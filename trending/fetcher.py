"""GitHub Trending repository fetcher using the GitHub Trending page.

This module fetches trending repositories from https://github.com/trending
by parsing the HTML page. This aligns with the task requirement to get repos
from the GitHub Trending page itself, not an approximation via GraphQL search.

The implementation bypasses initial filters by fetching the default trending
page without applying custom URL parameters or post-fetch filtering.
"""

from __future__ import annotations

import html
import logging
import re
import time
from typing import Any

import requests
from requests.exceptions import RequestException

import trending.config as config
from .logger import get_logger

logger = get_logger(__name__)


# GitHub Trending page URL
GITHUB_TRENDING_URL = "https://github.com/trending"


class TrendingFetcher:
    """Fetches trending repositories from the GitHub Trending page.

    This class fetches repositories from https://github.com/trending by parsing
    the HTML page. It bypasses initial filters by fetching the default trending
    page without applying custom URL parameters or post-fetch filtering.

    The implementation is modular with clear separation:
    - _fetch_trending_page: Fetches raw HTML from GitHub Trending
    - _parse_trending_html: Parses HTML to extract repository data
    - _normalize_repository: Normalizes parsed data to internal schema
    """

    def __init__(self) -> None:
        """Initialize the trending fetcher."""
        self.consecutive_failures = 0
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

    def _fetch_trending_page(self) -> str:
        """Fetch the raw HTML from the GitHub Trending page.

        Returns:
            Raw HTML content of the trending page.

        Raises:
            RequestException: If the HTTP request fails.
        """
        logger.info(f"Fetching trending page from {GITHUB_TRENDING_URL}")
        
        if config.GITHUB_MAX_RETRIES < 1:
            raise ValueError(f"GITHUB_MAX_RETRIES must be at least 1, got {config.GITHUB_MAX_RETRIES}")
        
        for attempt in range(config.GITHUB_MAX_RETRIES):
            try:
                response = self.session.get(
                    GITHUB_TRENDING_URL,
                    timeout=config.GITHUB_TIMEOUT_SECONDS,
                )
                response.raise_for_status()
                logger.info(f"Successfully fetched trending page (status: {response.status_code})")
                return response.text
            except RequestException as exc:
                if attempt == config.GITHUB_MAX_RETRIES - 1:
                    logger.error(f"Failed to fetch trending page after {config.GITHUB_MAX_RETRIES} attempts: {exc}")
                    raise
                logger.warning(f"Attempt {attempt + 1} failed, retrying: {exc}")
                time.sleep(2 ** attempt)

    def _parse_trending_html(self, raw_html: str) -> list[dict[str, Any]]:
        """Parse the GitHub Trending HTML to extract repository data.

        Args:
            raw_html: Raw HTML content from the trending page.

        Returns:
            List of parsed repository dictionaries with raw data from the page.
        """
        repositories = []
        
        # GitHub Trending page structure: each repo is in an article tag with class="Box-row"
        # We'll use regex to extract repo data from the HTML
        # This is a simplified parser that extracts the key information needed
        
        # Pattern to match repository entries
        # The structure typically includes: repo name, description, stars, forks, language, etc.
        repo_pattern = re.compile(
            r'<article[^>]*class="[^"]*Box-row[^"]*"[^>]*>.*?</article>',
            re.DOTALL
        )
        
        for match in repo_pattern.finditer(raw_html):
            repo_html = match.group(0)
            
            try:
                # Extract repo name/owner - look for the first link that looks like a repo
                name_match = re.search(r'href="/([^/]+)/([^/]+?)"', repo_html)
                if name_match:
                    owner = name_match.group(1)
                    name = name_match.group(2)
                    # Clean up name - it might have extra characters
                    name = re.sub(r'["\'].*$', '', name)
                    full_name = f"{owner}/{name}"
                    url = f"https://github.com/{full_name}"
                else:
                    continue
                
                # Extract description - look for paragraph with description
                desc_match = re.search(r'<p[^>]*class="[^"]*col-9[^"]*"[^>]*>(.*?)</p>', repo_html, re.DOTALL)
                description = desc_match.group(1).strip() if desc_match else ""
                # Clean up HTML tags from description
                description = re.sub(r'<[^>]+>', '', description)
                description = re.sub(r'\s+', ' ', description).strip()
                # Unescape HTML entities (e.g., R&amp;D -> R&D)
                description = html.unescape(description)
                
                # Extract star count - try stargazers link first, fallback to aria-label
                stars_match = re.search(r'<a[^>]*href=\"[^\"]*/stargazers[^\"]*\"[^>]*>.*?</svg>\s*([\d,]+)', repo_html, re.DOTALL)
                if not stars_match:
                    stars_match = re.search(r'<span[^>]*aria-label=\"star\"[^>]*>([\d,]+)', repo_html)
                star_count = int(stars_match.group(1).replace(',', '')) if stars_match else 0
                
                # Extract daily stars (stars today) if available
                daily_stars_match = re.search(r'([\d,]+)\s+stars\s+today', repo_html, re.IGNORECASE)
                daily_stars = int(daily_stars_match.group(1).replace(',', '')) if daily_stars_match else 0
                
                # Extract fork count - try forks link first, fallback to aria-label
                forks_match = re.search(r'<a[^>]*href=\"[^\"]*/forks[^\"]*\"[^>]*>.*?</svg>\s*([\d,]+)', repo_html, re.DOTALL)
                if not forks_match:
                    forks_match = re.search(r'<span[^>]*aria-label=\"fork\"[^>]*>([\d,]+)', repo_html)
                fork_count = int(forks_match.group(1).replace(',', '')) if forks_match else 0
                
                # Extract language - look for itemprop="programmingLanguage"
                lang_match = re.search(r'itemprop="programmingLanguage">([^<]+)', repo_html)
                primary_language = lang_match.group(1).strip() if lang_match else "Unknown"
                
                repositories.append({
                    "full_name": full_name,
                    "name": name,
                    "owner": owner,
                    "url": url,
                    "description": description,
                    "star_count": star_count,  # Total Lifetime Stars
                    "daily_stars": daily_stars,  # Stars gained today
                    "fork_count": fork_count,
                    "primary_language": primary_language,
                    # These fields will be filled in during normalization if needed
                    "created_at": "",
                    "pushed_at": "",
                    "topics": [],
                    "readme": "",
                    "default_branch": "main",
                })
                
            except Exception as exc:
                logger.warning(f"Failed to parse repository entry: {exc}")
                continue
        
        logger.info(f"Parsed {len(repositories)} repositories from trending page")
        return repositories

    def _normalize_repository(self, repo: dict[str, Any]) -> dict[str, Any]:
        """Normalize a parsed repository dictionary to the standard format.

        Args:
            repo: Parsed repository dictionary from HTML parsing.

        Returns:
            Normalized repository dictionary.
        """
        self.consecutive_failures = 0  # Reset on successful normalization
        
        # The parsed data from HTML already has most fields in the right format
        # We just need to ensure consistency and handle any missing fields
        return {
            "full_name": repo.get("full_name", ""),
            "name": repo.get("name", ""),
            "owner": repo.get("owner", ""),
            "url": repo.get("url", ""),
            "description": repo.get("description", "") or "",
            "star_count": repo.get("star_count", 0),
            "daily_stars": repo.get("daily_stars", 0),
            "fork_count": repo.get("fork_count", 0),
            "created_at": repo.get("created_at", "") or "",
            "pushed_at": repo.get("pushed_at", "") or "",
            "primary_language": repo.get("primary_language") or "Unknown",
            "topics": repo.get("topics", []),
            "readme": repo.get("readme", ""),
            "default_branch": repo.get("default_branch", "main"),
        }

    def fetch_trending_repositories(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Fetch trending repositories from GitHub Trending page.

        This method:
        1. Fetches the HTML from https://github.com/trending
        2. Parses the HTML to extract repository data
        3. Normalizes the data to the standard format
        4. Limits to the requested number of repositories

        Args:
            limit: Maximum number of repositories to fetch.
                Defaults to TRENDING_REPO_LIMIT from config.

        Returns:
            List of normalized repository dictionaries.

        Raises:
            RequestException: If the HTTP request fails.
        """
        target_limit = limit or config.TRENDING_REPO_LIMIT
        logger.info(f"Fetching up to {target_limit} trending repositories from GitHub Trending page")

        try:
            # Step 1: Fetch the trending page HTML
            html = self._fetch_trending_page()

            # Step 2: Parse the HTML to extract repository data
            parsed_repos = self._parse_trending_html(html)

            if not parsed_repos:
                logger.warning("No repositories parsed from trending page")
                return []

            # Step 3: Limit to the requested number
            parsed_repos = parsed_repos[:target_limit]
            logger.info(f"Limited to {len(parsed_repos)} repositories (requested: {target_limit})")

            # Step 4: Normalize the repositories
            repositories = []
            for repo in parsed_repos:
                try:
                    normalized = self._normalize_repository(repo)
                    repositories.append(normalized)
                    self.consecutive_failures = 0  # Reset on success
                except Exception as exc:
                    self.consecutive_failures += 1
                    logger.error(f"Failed to normalize repository: {exc}")
                    if not config.CONTINUE_ON_ERROR:
                        raise
                    if self.consecutive_failures >= config.MAX_CONSECUTIVE_FAILURES:
                        logger.error(
                            f"Max consecutive failures ({config.MAX_CONSECUTIVE_FAILURES}) reached. Stopping."
                        )
                        break

            logger.info(f"Successfully normalized {len(repositories)} repositories")
            return repositories

        except RequestException as exc:
            logger.error(f"Failed to fetch trending page: {exc}")
            raise
        except Exception as exc:
            logger.error(f"Unexpected error fetching trending repositories: {exc}")
            raise
