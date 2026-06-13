"""
gh-social-ml  ·  Full Acquisition → Feature Extraction → Scoring → Ranking Pipeline
======================================================================================

Architecture (per design diagram):
  [Acquisition]         Discovery  →  Enrichment
  [Feature Extraction]  Embeddings  →  Code Health Score  →  Doc Quality Score
  [Scoring]             Policy scoring  →  Batch scoring
  [Ranking / Output]    Final ranked results  →  Downstream consumers

Each stage is importable as a self-contained module.  Today, Acquisition is
fully implemented.  Everything downstream is wired up with stub interfaces so
the team can fill in each stage independently without touching this file.

Usage:
    python3 main.py [--limit N] [--batch-size N] [--log-level LEVEL]

Environment:
    GITHUB_TOKEN  — required, set in .env
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv

load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────

def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

logger = logging.getLogger("pipeline")


# ── Pipeline Result Container ─────────────────────────────────────────────────

@dataclass
class PipelineResult:
    """Carries data between pipeline stages."""

    # Stage 1 — Acquisition
    enriched_repos: list[Any] = field(default_factory=list)   # list[EnrichmentResult]

    # Stage 2 — Feature Extraction  (filled by feature_extraction stage)
    embeddings: dict[str, list[float]] = field(default_factory=dict)   # repo_id → vector
    code_health_scores: dict[str, float] = field(default_factory=dict) # repo_id → 0-1
    doc_quality_scores: dict[str, float] = field(default_factory=dict) # repo_id → 0-1

    # Stage 3 — Scoring  (filled by scoring stage)
    policy_scores: dict[str, float] = field(default_factory=dict)      # repo_id → score
    batch_scores: dict[str, float] = field(default_factory=dict)       # repo_id → score

    # Stage 4 — Ranking  (filled by ranking stage)
    ranked_repos: list[dict[str, Any]] = field(default_factory=list)   # final ordered list


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 1 — ACQUISITION
#  Status: ✅ Implemented
# ══════════════════════════════════════════════════════════════════════════════

def run_acquisition(
    token: str,
    *,
    limit: int = 100,
    batch_size: int = 10,
) -> list[Any]:
    """
    Discover and enrich GitHub repositories via GraphQL.

    Returns a list of EnrichmentResult objects. Each result carries:
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

    logger.info("─" * 60)
    logger.info("STAGE 1 · Acquisition  (target: %d repos, batch: %d)", limit, batch_size)
    logger.info("─" * 60)

    client = GitHubGraphQLClient(token=token)
    config = DiscoveryConfig(total_limit=limit + 20)   # small buffer so we hit the target
    discovery = GitHubDiscoveryEngine(client, config=config)
    enricher = RepositoryEnricher(graphql_client=client)

    # ── Discovery ─────────────────────────────────────────────────────────────
    logger.info("Discovering repositories …")
    discovered = discovery.discover(limit=limit + 20)
    logger.info("Discovered %d candidate repos", len(discovered))

    # ── Enrichment in batches ─────────────────────────────────────────────────
    logger.info("Enriching in batches of %d …", batch_size)
    enriched: list[Any] = []
    targets = discovered[:limit]
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

    logger.info("Acquisition complete — %d/%d repos enriched", len(enriched), limit)
    return enriched


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 2 — FEATURE EXTRACTION
#  Status: 🔲 Stub — wire in feature/embeddings.py, feature/code_health.py,
#                     feature/doc_quality.py when ready
# ══════════════════════════════════════════════════════════════════════════════

def run_feature_extraction(result: PipelineResult) -> None:
    """
    Populate result.embeddings, result.code_health_scores, result.doc_quality_scores.

    Planned sub-steps:
      1. Embedding model (sentence-transformers / text-embedding-3-small)
         Input  → repo readme clean_text + description + topics
         Output → 768-dim or 1536-dim float vector per repo

      2. Code Health Score  (0.0 – 1.0)
         Signals → commit_frequency, open_issues_ratio, has_ci, license_present,
                    pushed_days_ago, fork_count, star_velocity

      3. Doc Quality Score  (0.0 – 1.0)
         Signals → readme_length, extracted_paragraphs count,
                    readme_to_codebase_ratio, has_code_examples, has_badges

    Replace the stub body below once each sub-module is implemented:

        from feature.embeddings import EmbeddingModel
        from feature.code_health import CodeHealthScorer
        from feature.doc_quality import DocQualityScorer

        embedder = EmbeddingModel()
        health   = CodeHealthScorer()
        doc      = DocQualityScorer()

        for r in result.enriched_repos:
            result.embeddings[r.repo_id]          = embedder.encode(r)
            result.code_health_scores[r.repo_id]  = health.score(r)
            result.doc_quality_scores[r.repo_id]  = doc.score(r)
    """
    logger.info("─" * 60)
    logger.info("STAGE 2 · Feature Extraction  [STUB — not yet implemented]")
    logger.info("─" * 60)

    # Placeholder: zero-fill so downstream stages don't crash
    for r in result.enriched_repos:
        result.embeddings[r.repo_id]         = []    # replace with real vector
        result.code_health_scores[r.repo_id] = 0.0   # replace with real score
        result.doc_quality_scores[r.repo_id] = 0.0   # replace with real score

    logger.info(
        "Feature extraction stub ran for %d repos (all zeroed — implement stage)",
        len(result.enriched_repos),
    )


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 3 — SCORING
#  Status: 🔲 Stub — wire in scoring/policy.py, scoring/batch_scorer.py
# ══════════════════════════════════════════════════════════════════════════════

def run_scoring(result: PipelineResult) -> None:
    """
    Compute policy_scores and batch_scores for every enriched repo.

    Planned sub-steps:
      1. Policy Scoring
         Applies rule-based filters and boosts:
         - Boost: recently active (pushed_days_ago < 7), high star velocity
         - Penalty: archived, stale (pushed_days_ago > 365), no license
         - Hard filter: star_count < threshold

      2. Batch Scoring
         Runs all repos through a lightweight ML model (e.g. LightGBM / linear)
         trained on engagement signals, combining:
         - code_health_score  (from Stage 2)
         - doc_quality_score  (from Stage 2)
         - star deltas (delta_3d, delta_7d, delta_30d from payload)
         - pushed_days_ago
         Output → composite float score per repo

    Replace the stub body once scoring modules are ready:

        from scoring.policy import PolicyScorer
        from scoring.batch_scorer import BatchScorer

        policy  = PolicyScorer()
        batch   = BatchScorer.load("models/batch_scorer_v1.pkl")

        for r in result.enriched_repos:
            result.policy_scores[r.repo_id] = policy.score(r)
            result.batch_scores[r.repo_id]  = batch.score(r)
    """
    logger.info("─" * 60)
    logger.info("STAGE 3 · Scoring  [STUB — not yet implemented]")
    logger.info("─" * 60)

    for r in result.enriched_repos:
        p = r.payload
        # Minimal heuristic so ranking has something to sort on today
        naive = (
            p.get("star_count", 0) * 0.5
            + p.get("delta_7d", 0) * 10
            - p.get("pushed_days_ago", 999) * 0.1
        )
        result.policy_scores[r.repo_id] = max(naive, 0.0)
        result.batch_scores[r.repo_id]  = 0.0   # replace with real ML score

    logger.info("Scoring stub ran — naive heuristic scores computed")


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 4 — RANKING & OUTPUT
#  Status: 🔲 Stub — wire in ranking/ranker.py, output/sink.py
# ══════════════════════════════════════════════════════════════════════════════

def run_ranking(result: PipelineResult) -> None:
    """
    Produce result.ranked_repos — the final sorted list ready for consumers.

    Planned sub-steps:
      1. Combine policy_score + batch_score into a final composite score.
      2. Apply diversity re-ranking (MMR or DPP) to avoid topic clustering.
      3. Emit ranked list to downstream sinks:
         - Qdrant vector store (for similarity search)
         - PostgreSQL / Firestore (for feed serving)
         - Webhook / pub-sub topic (for real-time push)

    Replace the stub body:

        from ranking.ranker import Ranker
        from output.sink import QdrantSink, DatabaseSink

        ranker = Ranker()
        result.ranked_repos = ranker.rank(result)

        QdrantSink().write(result)
        DatabaseSink().write(result)
    """
    logger.info("─" * 60)
    logger.info("STAGE 4 · Ranking & Output  [STUB — not yet implemented]")
    logger.info("─" * 60)

    def _composite(r: Any) -> float:
        pid = r.repo_id
        return result.policy_scores.get(pid, 0.0) + result.batch_scores.get(pid, 0.0)

    sorted_repos = sorted(result.enriched_repos, key=_composite, reverse=True)
    result.ranked_repos = [
        {
            "rank":             i + 1,
            "repo_id":          r.repo_id,
            "composite_score":  round(_composite(r), 4),
            "policy_score":     round(result.policy_scores.get(r.repo_id, 0.0), 4),
            "batch_score":      round(result.batch_scores.get(r.repo_id, 0.0), 4),
            **r.payload,
        }
        for i, r in enumerate(sorted_repos)
    ]

    logger.info("Ranking complete — %d repos ordered by composite score", len(result.ranked_repos))


# ══════════════════════════════════════════════════════════════════════════════
#  SUMMARY PRINTER
# ══════════════════════════════════════════════════════════════════════════════

def _print_summary(result: PipelineResult) -> None:
    repos = result.ranked_repos
    if not repos:
        logger.warning("No ranked repos to display.")
        return

    width = 100
    print(f"\n{'═' * width}")
    print(f"  Pipeline complete — Top {min(len(repos), 20)} / {len(repos)} repos")
    print(f"{'═' * width}")
    print(
        f"{'#':<4} {'Repository':<42} {'⭐':>7} {'Score':>8} "
        f"{'Language':<14} {'README':>8}  Topics"
    )
    print("─" * width)

    for row in repos[:20]:
        topics_str = ", ".join(row.get("topics", [])[:3]) or "—"
        print(
            f"{row['rank']:<4} {row['repo_id']:<42} {row.get('star_count', 0):>7,} "
            f"{row['composite_score']:>8.2f} {row.get('primary_language', 'Unknown'):<14} "
            f"{row.get('readme_length', 0):>7,}c  {topics_str}"
        )

    print(f"{'═' * width}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(
    token: str,
    *,
    limit: int = 100,
    batch_size: int = 10,
    log_level: str = "INFO",
) -> PipelineResult:
    """Execute all pipeline stages in order and return the final PipelineResult."""

    _setup_logging(log_level)

    logger.info("╔══════════════════════════════════════════════╗")
    logger.info("║     gh-social-ml  Acquisition Pipeline       ║")
    logger.info("╚══════════════════════════════════════════════╝")

    result = PipelineResult()

    # ── Stage 1: Acquisition ──────────────────────────────────────────────────
    result.enriched_repos = run_acquisition(token, limit=limit, batch_size=batch_size)

    if not result.enriched_repos:
        logger.error("Acquisition returned 0 repos — aborting pipeline.")
        return result

    # ── Stage 2: Feature Extraction ───────────────────────────────────────────
    run_feature_extraction(result)

    # ── Stage 3: Scoring ──────────────────────────────────────────────────────
    run_scoring(result)

    # ── Stage 4: Ranking & Output ─────────────────────────────────────────────
    run_ranking(result)

    # ── Summary ───────────────────────────────────────────────────────────────
    _print_summary(result)

    return result


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="main.py",
        description="gh-social-ml full pipeline: Acquisition → Features → Scoring → Ranking",
    )
    p.add_argument("--limit",      type=int, default=100,   help="Target number of repos (default: 100)")
    p.add_argument("--batch-size", type=int, default=10,    help="Enrichment batch size (default: 10)")
    p.add_argument("--log-level",  type=str, default="INFO", help="Logging level (default: INFO)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    token = os.getenv("GITHUB_TOKEN")
    if not token or token == "your_github_token_here":
        print("❌  ERROR: Set GITHUB_TOKEN in your .env file first.")
        sys.exit(1)

    run_pipeline(
        token,
        limit=args.limit,
        batch_size=args.batch_size,
        log_level=args.log_level,
    )
