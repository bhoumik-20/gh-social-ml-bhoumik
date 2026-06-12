"""GitHub GraphQL API client for repository discovery and enrichment."""

from __future__ import annotations

import os
import random
import time
from datetime import datetime, timezone
from typing import Any

import requests

from .graphql_queries import GET_README_QUERY, GET_REPOSITORY_QUERY, build_batch_metadata_query
from .github_client import GitHubClientError, GitHubRateLimit


class GitHubGraphQLClient:
    def __init__(
        self,
        *,
        token: str | None = None,
        base_url: str = "https://api.github.com/graphql",
        timeout_seconds: float = 30.0,
        max_retries: int = 4,
        sleep_on_rate_limit: bool = True,
        session: requests.Session | None = None,
    ) -> None:
        self.url = base_url
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.sleep_on_rate_limit = sleep_on_rate_limit
        self.session = session or requests.Session()
        token = token if token is not None else os.getenv("GITHUB_TOKEN")
        self.session.headers.update(
            {
                "User-Agent": "osiris-repository-ingestion-pipeline",
                "Content-Type": "application/json",
            }
        )
        if token:
            self.session.headers["Authorization"] = f"Bearer {token}"

    def get_repository(self, owner: str, name: str) -> dict[str, Any] | None:
        """Fetches a single repository using GraphQL."""
        variables = {"owner": owner, "name": name}
        response = self.execute(GET_REPOSITORY_QUERY, variables)
        if not response:
            return None
        data = response.get("data", {})
        return data.get("repository")

    def get_readme(self, owner: str, name: str) -> str:
        """Fetches only the README text for a single repo. Returns empty string if none."""
        try:
            response = self.execute(GET_README_QUERY, {"owner": owner, "name": name})
            if not response:
                return ""
            repo = (response.get("data") or {}).get("repository") or {}
            for key in ["readme1", "readme2", "readme3", "readme4", "readme5"]:
                blob = repo.get(key)
                if blob and blob.get("text"):
                    return blob["text"]
        except Exception:
            pass
        return ""

    def get_repositories_batch(self, repos: list[tuple[str, str]]) -> dict[str, dict[str, Any]]:
        """Fetches multiple repositories using a lean metadata-only batch query."""
        if not repos:
            return {}

        query = build_batch_metadata_query(repos)
        response = self.execute(query)
        if not response:
            return {}

        data = response.get("data", {})
        results = {}
        for i, (owner, name) in enumerate(repos):
            alias = f"repo_{i}"
            if alias in data and data[alias]:
                results[f"{owner}/{name}"] = data[alias]
        return results

    def execute(self, query: str, variables: dict[str, Any] | None = None) -> dict[str, Any] | None:
        """Executes a GraphQL query with retries and rate limit handling."""
        payload = {"query": query}
        if variables:
            payload["variables"] = variables

        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.post(
                    self.url,
                    json=payload,
                    timeout=self.timeout_seconds,
                )
            except requests.RequestException as exc:
                if attempt >= self.max_retries:
                    raise GitHubClientError(f"GitHub GraphQL request failed: {exc}") from exc
                self._sleep_backoff(attempt)
                continue

            if response.status_code == 403 and self._is_rate_limited(response):
                if attempt >= self.max_retries or not self.sleep_on_rate_limit:
                    raise GitHubClientError("GitHub GraphQL rate limit exceeded")
                self._sleep_until_reset(response)
                continue

            if response.status_code in {500, 502, 503, 504}:
                if attempt >= self.max_retries:
                    raise GitHubClientError(f"GitHub GraphQL transient failure {response.status_code}: {response.text[:300]}")
                self._sleep_backoff(attempt)
                continue

            if response.status_code >= 400:
                raise GitHubClientError(f"GitHub GraphQL error {response.status_code}: {response.text[:300]}")

            result = response.json()
            
            # Rate limit tracking from GraphQL payload
            data_field = result.get("data")
            if isinstance(data_field, dict) and "rateLimit" in data_field:
                rl = data_field["rateLimit"]
                logger.debug("GraphQL rate limit: %s remaining", rl.get("remaining"))
            
            if "errors" in result:
                # Some errors are partial, e.g., missing repository
                # We check if there's actual data returned
                if not result.get("data"):
                    if any("Could not resolve to a Repository" in e.get("message", "") for e in result["errors"]):
                        return None
                    raise GitHubClientError(f"GitHub GraphQL returned errors: {result['errors']}")

            return result

        raise GitHubClientError("GitHub GraphQL request exhausted retries")

    @staticmethod
    def _is_rate_limited(response: requests.Response) -> bool:
        remaining = response.headers.get("X-RateLimit-Remaining")
        return remaining == "0" or "rate limit" in response.text.lower()

    def _sleep_until_reset(self, response: requests.Response) -> None:
        reset = response.headers.get("X-RateLimit-Reset")
        if reset and reset.isdigit():
            sleep_for = max(int(reset) - int(time.time()) + 2, 1)
        else:
            retry_after = response.headers.get("Retry-After")
            sleep_for = int(retry_after) if retry_after and retry_after.isdigit() else 60
        time.sleep(min(sleep_for, 300))

    @staticmethod
    def _sleep_backoff(attempt: int) -> None:
        time.sleep(min((2**attempt) + random.random(), 30.0))
