"""Acquisition pipeline logic for discovering and enriching GitHub repositories."""

from __future__ import annotations

import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

logger = logging.getLogger("pipeline.acquisition")


def run_acquisition(
    token: str,
    *,
    limit: int = 150,
    batch_size: int = 15,
    workers: int = 4,
    existing_repos: set[str] | None = None,
) -> list[Any]:
    """
    Discover and enrich GitHub repositories via GraphQL only.

    Returns a list of EnrichmentResult objects. Each carries:
      .repo_id          — "owner/repo"
      .payload          — Osiris-compatible dict (star_count, language, topics, …)
      .raw_repository   — raw GraphQL response fields
      .readme           — ReadmeDocument (clean_text, extracted_paragraphs, …)
      .topics           — list[str]
      .languages        — dict[str, int]  (language → bytes)
    """
    # Clamp workers so ThreadPoolExecutor never receives 0 or a negative value,
    # which would raise ValueError even when called outside of the CLI.
    workers = max(1, workers)

    from acquisition.github_graphql_client import GitHubGraphQLClient
    from acquisition.github_discovery import GitHubDiscoveryEngine, DiscoveryConfig
    from acquisition.repository_enricher import RepositoryEnricher

    client   = GitHubGraphQLClient(token=token)
    # Fetch a larger buffer of candidate repositories to account for filtering duplicates
    discovery_limit = limit + 50 if existing_repos else limit + 20
    config   = DiscoveryConfig(total_limit=discovery_limit)
    discovery = GitHubDiscoveryEngine(client, config=config)
    enricher  = RepositoryEnricher(graphql_client=client)

    # ── Step 1: Discovery ─────────────────────────────────────────────────────
    logger.info("Discovering repositories …")
    discovered = discovery.discover(limit=discovery_limit)
    logger.info("Discovered %d candidate repos", len(discovered))

    if existing_repos:
        new_discovered = []
        for r in discovered:
            full_name = r if isinstance(r, str) else r.get("full_name", "")
            if full_name not in existing_repos:
                new_discovered.append(r)
        logger.info(
            "Filtered out %d already existing repos from candidates. %d new candidates remain.",
            len(discovered) - len(new_discovered),
            len(new_discovered),
        )
        discovered = new_discovered

    # ── Step 2: Concurrent enrichment ─────────────────────────────────────────
    targets = discovered[:limit]
    batches = [targets[i : i + batch_size] for i in range(0, len(targets), batch_size)]
    logger.info(
        "Enriching %d repos in %d batch(es) of up to %d with %d concurrent worker(s) …",
        len(targets), len(batches), batch_size, workers,
    )
    enriched: list = []

    # Each worker thread gets its own GitHubGraphQLClient (and requests.Session)
    # so concurrent threads never race on a shared Session object.
    _thread_local = threading.local()

    def _get_thread_enricher() -> "RepositoryEnricher":
        if not hasattr(_thread_local, "enricher"):
            from acquisition.github_graphql_client import GitHubGraphQLClient
            from acquisition.repository_enricher import RepositoryEnricher
            _thread_local.enricher = RepositoryEnricher(
                graphql_client=GitHubGraphQLClient(token=token)
            )
        return _thread_local.enricher

    def _enrich_batch(batch: list[Any]) -> list[Any]:
        # get_repositories_batch() uses a two-pass approach: lean metadata first,
        # README fetched separately. A README failure only produces empty README
        # data; the repo is still kept, not dropped entirely.
        return _get_thread_enricher().get_repositories_batch(batch)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Each future processes one batch of batch_size repos. workers controls
        # how many batches run concurrently; batch_size controls the GraphQL
        # request payload size. Both levers are now effective.
        futures = {
            executor.submit(_enrich_batch, batch): (i + 1, batch)
            for i, batch in enumerate(batches)
        }
        for future in as_completed(futures):
            batch_num, batch = futures[future]
            try:
                results = future.result()
                enriched.extend(results)
                logger.info(
                    "  Batch %d/%d → +%d enriched  (total: %d)",
                    batch_num, len(batches), len(results), len(enriched),
                )
            except Exception as exc:
                logger.warning("  Batch %d/%d failed: %s", batch_num, len(batches), exc)


    logger.info("Acquisition complete — %d / %d repos enriched", len(enriched), limit)
    return enriched

