"""Balanced ecosystem discovery for GitHub repositories."""

from __future__ import annotations

import itertools
import logging
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from .github_graphql_client import GitHubGraphQLClient
from .graphql_queries import SEARCH_REPOSITORIES_QUERY

logger = logging.getLogger(__name__)


DISCOVERY_CATEGORIES: dict[str, list[str]] = {
    "AI": ["artificial intelligence", "ai"],
    "LLM": ["llm", "large language model"],
    "RAG": ["rag", "retrieval augmented generation"],
    "AI Agents": ["ai agent", "multi-agent"],
    "Frontend": ["frontend", "ui components"],
    "Backend": ["backend", "api server"],
    "Developer Tools": ["developer tools", "cli"],
    "Web Frameworks": ["web framework", "http framework"],
    "Security": ["security", "vulnerability"],
    "Infrastructure": ["infrastructure", "kubernetes"],
    "Databases": ["database", "storage engine"],
    "Observability": ["observability", "monitoring"],
    "DevOps": ["devops", "ci cd"],
    "Cloud": ["cloud native", "serverless"],
    "Automation": ["automation", "workflow"],
    "ML": ["machine learning", "ml"],
    "Computer Vision": ["computer vision", "image recognition"],
    "Robotics": ["robotics", "robot"],
    "Bioinformatics": ["bioinformatics", "genomics"],
    "Embedded Systems": ["embedded", "firmware"],
    "Systems Programming": ["systems programming", "kernel"],
    "Mobile": ["mobile app", "android ios"],
    "Game Development": ["game development", "game engine"],
}


@dataclass(slots=True)
class DiscoveryConfig:
    total_limit: int = 120
    per_query: int = 20
    pages_per_query: int = 1
    random_seed: int | None = None


class GitHubDiscoveryEngine:
    """Discovers repositories across categories and maturity bands using GraphQL only."""

    def __init__(self, client: GitHubGraphQLClient, *, config: DiscoveryConfig | None = None) -> None:
        self.client = client
        self.config = config or DiscoveryConfig()
        self.random = random.Random(self.config.random_seed)

    def _search_graphql(self, query_str: str, per_page: int) -> list[dict[str, Any]]:
        """Run a GraphQL repository search and return a flat list of repo dicts."""
        result = self.client.execute(
            SEARCH_REPOSITORIES_QUERY,
            {"query": query_str},
        )
        if not result:
            return []
        nodes = result.get("data", {}).get("search", {}).get("nodes", [])
        repos = []
        for node in nodes[:per_page]:
            full_name = node.get("nameWithOwner")
            if not full_name:
                continue
            repos.append({
                "full_name": full_name,
                "name": node.get("name"),
                "stargazers_count": node.get("stargazerCount", 0),
                "description": node.get("description"),
                "pushed_at": node.get("pushedAt"),
                "owner": {"login": (node.get("owner") or {}).get("login")},
            })
        return repos

    def discover(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        target = limit or self.config.total_limit
        allocations = self._allocations(target)
        discovered: dict[str, dict[str, Any]] = {}

        category_cycle = list(DISCOVERY_CATEGORIES.items())
        self.random.shuffle(category_cycle)

        for band, band_limit in allocations.items():
            if len(discovered) >= target:
                break
            per_category = max(1, band_limit // max(len(category_cycle), 1))
            for category, terms in category_cycle:
                if len(discovered) >= target:
                    break
                query = self._query_for_band(category, terms, band)
                per_page = min(self.config.per_query, max(per_category * 2, 10))
                try:
                    repos = self._search_graphql(query, per_page)
                except Exception as exc:
                    logger.warning(f"Discovery query failed for {category}/{band}: {exc}")
                    continue
                for repo in repos:
                    full_name = repo.get("full_name")
                    if not full_name or full_name in discovered:
                        continue
                    repo["_discovery_category"] = category
                    repo["_discovery_band"] = band
                    discovered[full_name] = repo
                    if len([r for r in discovered.values() if r.get("_discovery_band") == band]) >= band_limit:
                        break

        return list(discovered.values())[:target]

    @staticmethod
    def _allocations(total: int) -> dict[str, int]:
        high = int(total * 0.40)
        recent = int(total * 0.30)
        mid = int(total * 0.20)
        emerging = total - high - recent - mid
        return {
            "high_star": high,
            "recently_active": recent,
            "mid_sized": mid,
            "emerging": emerging,
        }

    def _query_for_band(self, category: str, terms: list[str], band: str) -> str:
        term = self.random.choice(terms)
        now = datetime.now(timezone.utc)
        pushed_recent = (now - timedelta(days=45)).date().isoformat()
        created_recent = (now - timedelta(days=180)).date().isoformat()
        # Randomize search query to get new repositories instead of duplicate top results
        rand_char = self.random.choice("abcdefghijklmnopqrstuvwxyz")
        base = f'{term} {rand_char} in:name,description,readme fork:false archived:false'
        if band == "high_star":
            return f"{base} stars:>500"
        if band == "recently_active":
            return f"{base} pushed:>{pushed_recent} stars:50..5000"
        if band == "mid_sized":
            return f"{base} stars:50..500"
        return f"{base} created:>{created_recent} stars:5..150"
