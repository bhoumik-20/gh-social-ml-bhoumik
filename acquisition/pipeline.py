"""Acquisition pipeline logic for discovering and enriching GitHub repositories."""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger("pipeline.acquisition")


def run_acquisition(
    token: str,
    *,
    limit: int = 100,
    batch_size: int = 10,
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

    # ── Step 2: Enrichment in batches ─────────────────────────────────────────
    logger.info("Enriching in batches of %d …", batch_size)
    enriched: list = []
    targets       = discovered[:limit]
    total_batches = (len(targets) + batch_size - 1) // batch_size

    for i in range(total_batches):
        batch = targets[i * batch_size : (i + 1) * batch_size]
        try:
            results = enricher.get_repositories_batch(batch)
            enriched.extend(results)
            logger.info(
                "  Batch %d/%d → +%d enriched  (total: %d)",
                i + 1, total_batches, len(results), len(enriched),
            )
        except Exception as exc:
            logger.warning("  Batch %d failed (%s). Falling back to one-by-one …", i + 1, exc)
            for repo in batch:
                full_name = repo if isinstance(repo, str) else repo.get("full_name", "")
                try:
                    r = enricher.enrich(full_name)
                    if r:
                        enriched.append(r)
                        logger.info("    ✓  %s", full_name)
                except Exception as exc2:
                    logger.warning("    ✗  %s: %s", full_name, exc2)

    logger.info("Acquisition complete — %d / %d repos enriched", len(enriched), limit)
    return enriched
