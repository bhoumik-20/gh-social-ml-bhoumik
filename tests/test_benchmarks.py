"""Performance benchmarks for trending service."""

from unittest.mock import MagicMock, patch
import pytest

from trending.fetcher import TrendingFetcher
from trending.storage import TrendingStorage


@pytest.mark.benchmark
class TestFetchPerformance:
    """Benchmark fetch operations for HTML-based Trending page implementation."""

    def test_fetch_30_repositories_performance(self, benchmark):
        """Benchmark fetching 30 repositories from GitHub Trending page."""
        fetcher = TrendingFetcher()
        
        def fetch_operation():
            with patch.object(fetcher, '_fetch_trending_page', return_value="<html>test</html>"):
                with patch.object(fetcher, '_parse_trending_html', return_value=[
                    {
                        "full_name": f"test-owner/test-repo{i}",
                        "name": f"test-repo{i}",
                        "owner": "test-owner",
                        "url": f"https://github.com/test-owner/test-repo{i}",
                        "description": f"Test repository {i}",
                        "star_count": 1000 + i,
                        "fork_count": 50 + i,
                        "created_at": "2024-01-01T00:00:00Z",
                        "pushed_at": "2024-06-01T00:00:00Z",
                        "primary_language": "Python",
                        "topics": [],
                        "readme": "",
                        "default_branch": "main",
                    }
                    for i in range(30)
                ]):
                    return fetcher.fetch_trending_repositories(limit=30)
        
        result = benchmark(fetch_operation)
        assert len(result) == 30

    def test_normalize_repository_performance(self, benchmark):
        """Benchmark repository normalization."""
        fetcher = TrendingFetcher()
        repo = {
            "full_name": "test-owner/test-repo",
            "name": "test-repo",
            "owner": "test-owner",
            "url": "https://github.com/test-owner/test-repo",
            "description": "A test repository",
            "star_count": 1000,
            "fork_count": 50,
            "created_at": "2024-01-01T00:00:00Z",
            "pushed_at": "2024-06-01T00:00:00Z",
            "primary_language": "Python",
            "topics": ["machine-learning", "data-science"],
            "readme": "",
            "default_branch": "main",
        }
        
        def normalize_operation():
            return fetcher._normalize_repository(repo)
        
        result = benchmark(normalize_operation)
        assert result["full_name"] == "test-owner/test-repo"
        assert result["star_count"] == 1000

    def test_parse_html_performance(self, benchmark):
        """Benchmark HTML parsing of trending page."""
        fetcher = TrendingFetcher()
        
        # Create a sample HTML with multiple repo entries
        sample_html = """
        <article class="Box-row">
            <a href="/test-owner/test-repo1">test-repo1</a>
            <p class="col-9">Test repository 1</p>
            <span aria-label="star">1000</span>
            <span aria-label="fork">50</span>
            <span itemprop="programmingLanguage">Python</span>
        </article>
        """ * 30
        
        def parse_operation():
            return fetcher._parse_trending_html(sample_html)
        
        result = benchmark(parse_operation)
        # The regex may not match perfectly with this simple HTML, but we're benchmarking the parsing logic
        assert isinstance(result, list)


@pytest.mark.benchmark
class TestStoragePerformance:
    """Benchmark storage operations."""

    def test_upsert_30_repositories_performance(self, benchmark, mock_database_connector, sample_normalized_repo):
        """Benchmark upserting 30 repositories."""
        storage = TrendingStorage()
        storage.connector = mock_database_connector
        storage.enabled = True
        
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_database_connector.connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.return_value = None
        
        repos = [sample_normalized_repo] * 30
        
        def upsert_operation():
            return storage.upsert_repositories(repos)
        
        result = benchmark(upsert_operation)
        assert result == 30

    def test_parse_timestamp_performance(self, benchmark):
        """Benchmark timestamp parsing."""
        timestamp = "2024-06-20T12:00:00Z"
        
        result = benchmark(TrendingStorage._parse_timestamp, timestamp)
        assert result is not None

    def test_get_trending_repositories_performance(self, benchmark, mock_database_connector):
        """Benchmark retrieving trending repositories."""
        storage = TrendingStorage()
        storage.connector = mock_database_connector
        storage.enabled = True
        
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_database_connector.connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchall.return_value = [
            ("test/repo", "repo", "test", "https://github.com/test/repo", "Test",
             100, 100, 10, "2024-01-01", "2024-06-01", "Python", '["ml", "ai"]',
             "# README", "main", "2024-01-01", "2024-06-01", i)
            for i in range(1, 31)
        ]
        
        result = benchmark(storage.get_trending_repositories, limit=30)
        assert len(result) == 30


@pytest.mark.benchmark
class TestConfigPerformance:
    """Benchmark configuration operations."""

    def test_config_validation_performance(self, benchmark):
        """Benchmark configuration validation."""
        from trending.config import validate_config
        
        result = benchmark(validate_config)
        assert isinstance(result, list)


@pytest.mark.benchmark
class TestLoggerPerformance:
    """Benchmark logging operations."""

    def test_logger_setup_performance(self, benchmark):
        """Benchmark logger setup."""
        from trending.logger import setup_logger
        
        result = benchmark(setup_logger, "benchmark_logger")
        assert result.name == "benchmark_logger"


@pytest.mark.benchmark
class TestEndToEndPerformance:
    """Benchmark end-to-end operations for HTML-based Trending page implementation."""

    def test_complete_fetch_normalize_cycle_performance(self, benchmark):
        """Benchmark complete fetch and normalize cycle for Trending page."""
        fetcher = TrendingFetcher()
        
        def complete_cycle():
            with patch.object(fetcher, '_fetch_trending_page', return_value="<html>test</html>"):
                with patch.object(fetcher, '_parse_trending_html', return_value=[
                    {
                        "full_name": f"test-owner/test-repo{i}",
                        "name": f"test-repo{i}",
                        "owner": "test-owner",
                        "url": f"https://github.com/test-owner/test-repo{i}",
                        "description": f"Test repository {i}",
                        "star_count": 1000 + i,
                        "fork_count": 50 + i,
                        "created_at": "2024-01-01T00:00:00Z",
                        "pushed_at": "2024-06-01T00:00:00Z",
                        "primary_language": "Python",
                        "topics": [],
                        "readme": "",
                        "default_branch": "main",
                    }
                    for i in range(30)
                ]):
                    repos = fetcher.fetch_trending_repositories(limit=30)
                    return len(repos)
        
        result = benchmark(complete_cycle)
        assert result == 30
