"""Integration tests for trending service end-to-end pipeline."""

from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from trending.scheduler import TrendingScheduler
from trending.fetcher import TrendingFetcher
from trending.storage import TrendingStorage
from requests.exceptions import RequestException


@pytest.mark.integration
class TestEndToEndPipeline:
    """Test complete end-to-end trending repository pipeline."""

    @patch('trending.fetcher.TrendingFetcher._parse_trending_html')
    @patch('trending.fetcher.TrendingFetcher._fetch_trending_page')
    def test_complete_fetch_and_store_cycle(self, mock_fetch, mock_parse, mock_database_connector):
        """Test complete pipeline from GitHub Trending page fetch to database storage."""
        # Mock HTML response
        mock_fetch.return_value = "<html>test</html>"
        
        # Mock parsed repositories
        mock_parse.return_value = [
            {
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
                "topics": [],
                "readme": "",
                "default_branch": "main",
            },
            {
                "full_name": "test-owner/test-repo2",
                "name": "test-repo2",
                "owner": "test-owner",
                "url": "https://github.com/test-owner/test-repo2",
                "description": "Another test repository",
                "star_count": 2000,
                "fork_count": 100,
                "created_at": "2024-01-02T00:00:00Z",
                "pushed_at": "2024-06-02T00:00:00Z",
                "primary_language": "JavaScript",
                "topics": [],
                "readme": "",
                "default_branch": "main",
            },
            {
                "full_name": "test-owner/test-repo3",
                "name": "test-repo3",
                "owner": "test-owner",
                "url": "https://github.com/test-owner/test-repo3",
                "description": "Third test repository",
                "star_count": 3000,
                "fork_count": 150,
                "created_at": "2024-01-03T00:00:00Z",
                "pushed_at": "2024-06-03T00:00:00Z",
                "primary_language": "TypeScript",
                "topics": [],
                "readme": "",
                "default_branch": "main",
            },
        ]
        
        mock_database_connector.enabled = True
        
        # Create real instances with mocked dependencies
        fetcher = TrendingFetcher()
        storage = TrendingStorage()
        storage.connector = mock_database_connector
        storage.enabled = True
        
        # Mock database operations
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_database_connector.connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.return_value = None  # New repositories
        
        # Execute fetch
        repositories = fetcher.fetch_trending_repositories(limit=3)
        
        assert len(repositories) == 3
        assert all("full_name" in repo for repo in repositories)
        assert all("star_count" in repo for repo in repositories)
        
        # Execute storage
        with patch.object(storage, 'init_schema'):
            upserted_count = storage.upsert_repositories(repositories)
        
        assert upserted_count == 3

    @patch('trending.fetcher.TrendingFetcher._parse_trending_html')
    @patch('trending.fetcher.TrendingFetcher._fetch_trending_page')
    def test_pipeline_idempotent_upsert(self, mock_fetch, mock_parse, mock_database_connector):
        """Test that pipeline handles duplicate repository upserts correctly."""
        mock_fetch.return_value = "<html>test</html>"
        mock_parse.return_value = [
            {
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
                "topics": [],
                "readme": "",
                "default_branch": "main",
            },
        ]
        
        mock_database_connector.enabled = True
        
        fetcher = TrendingFetcher()
        storage = TrendingStorage()
        storage.connector = mock_database_connector
        storage.enabled = True
        
        # Mock database operations
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_database_connector.connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        
        # First upsert - repository doesn't exist
        mock_cursor.fetchone.return_value = None
        repositories = fetcher.fetch_trending_repositories(limit=1)
        
        with patch.object(storage, 'init_schema'):
            first_upsert = storage.upsert_repositories(repositories)
        
        assert first_upsert == 1
        
        # Second upsert - repository exists
        from datetime import datetime, timezone
        mock_cursor.fetchone.return_value = (datetime.now(timezone.utc),)
        
        with patch.object(storage, 'init_schema'):
            second_upsert = storage.upsert_repositories(repositories)
        
        assert second_upsert == 1  # Should still succeed (update)

    @patch('trending.fetcher.TrendingFetcher._parse_trending_html')
    @patch('trending.fetcher.TrendingFetcher._fetch_trending_page')
    def test_pipeline_metadata_tracking(self, mock_fetch, mock_parse, mock_database_connector):
        """Test that pipeline correctly tracks metadata."""
        mock_fetch.return_value = "<html>test</html>"
        mock_parse.return_value = [
            {
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
                "topics": [],
                "readme": "",
                "default_branch": "main",
            },
        ]
        
        mock_database_connector.enabled = True
        
        fetcher = TrendingFetcher()
        storage = TrendingStorage()
        storage.connector = mock_database_connector
        storage.enabled = True
        
        # Mock database operations
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_database_connector.connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.return_value = None
        
        repositories = fetcher.fetch_trending_repositories(limit=1)
        
        with patch.object(storage, 'init_schema'):
            storage.upsert_repositories(repositories)
        
        # Verify metadata update calls
        assert mock_cursor.execute.call_count >= 2  # At least upsert + metadata updates

    def test_pipeline_with_empty_response(self, mock_database_connector):
        """Test pipeline handles empty response gracefully."""
        mock_database_connector.enabled = True
        
        fetcher = TrendingFetcher()
        storage = TrendingStorage()
        storage.connector = mock_database_connector
        storage.enabled = True
        
        # Mock the methods at instance level
        with patch.object(fetcher, '_fetch_trending_page', return_value="<html>test</html>"):
            with patch.object(fetcher, '_parse_trending_html', return_value=[]):
                repositories = fetcher.fetch_trending_repositories()
                
                assert repositories == []
        
        # Should handle empty list gracefully
        with patch.object(storage, 'init_schema'):
            upserted = storage.upsert_repositories(repositories)
        
        assert upserted == 0

    def test_pipeline_with_network_error(self, mock_database_connector):
        """Test pipeline handles network errors gracefully."""
        from requests.exceptions import RequestException
        mock_database_connector.enabled = True
        
        fetcher = TrendingFetcher()
        
        # Mock the session.get to raise RequestException
        with patch.object(fetcher.session, 'get', side_effect=RequestException("Network error")):
            # The fetcher raises RequestException after all retries are exhausted
            with pytest.raises(RequestException):
                fetcher.fetch_trending_repositories()

    @patch('trending.fetcher.TrendingFetcher._parse_trending_html')
    @patch('trending.fetcher.TrendingFetcher._fetch_trending_page')
    def test_scheduler_integration(self, mock_fetch, mock_parse, mock_database_connector):
        """Test scheduler integration with fetch and storage."""
        import os
        import importlib
        os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost:5432/test_db")
        
        # Reload config to pick up the environment variable
        import trending.config
        importlib.reload(trending.config)
        
        mock_fetch.return_value = "<html>test</html>"
        mock_parse.return_value = [
            {
                "full_name": "test/repo",
                "name": "repo",
                "owner": "test",
                "url": "https://github.com/test/repo",
                "description": "",
                "star_count": 100,
                "fork_count": 0,
                "created_at": "",
                "pushed_at": "",
                "primary_language": "Python",
                "topics": [],
                "readme": "",
                "default_branch": "main"
            }
        ]
        
        mock_database_connector.enabled = True
        
        fetcher = TrendingFetcher()
        storage = TrendingStorage()
        storage.connector = mock_database_connector
        storage.enabled = True
        
        # Mock database operations
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_database_connector.connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.return_value = None
        
        scheduler = TrendingScheduler(fetcher=fetcher, storage=storage)
        
        with patch.object(storage, 'init_schema'):
            result = scheduler.refresh_trending_repositories()
        
        assert result is True


@pytest.mark.integration
class TestConfigurationLoading:
    """Test configuration loading in integration context."""

    def test_load_dotenv_in_integration(self, monkeypatch):
        """Test that load_dotenv works in integration context."""
        # Set environment variables
        monkeypatch.setenv("DATABASE_URL", "postgresql://test:test@localhost:5432/test_db")
        
        # Reload config to pick up new environment
        import importlib
        import trending.config
        importlib.reload(trending.config)
        
        # Verify configuration loaded
        assert trending.config.DATABASE_URL == "postgresql://test:test@localhost:5432/test_db"


@pytest.mark.integration
class TestErrorRecovery:
    """Test error recovery in integration scenarios."""

    @patch('trending.fetcher.TrendingFetcher._parse_trending_html')
    @patch('trending.fetcher.TrendingFetcher._fetch_trending_page')
    def test_partial_failure_recovery(self, mock_fetch, mock_parse, mock_database_connector):
        """Test pipeline recovers from partial failures."""
        mock_fetch.return_value = "<html>test</html>"
        mock_parse.return_value = [
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
            for i in range(3)
        ]
        
        mock_database_connector.enabled = True
        
        fetcher = TrendingFetcher()
        storage = TrendingStorage()
        storage.connector = mock_database_connector
        storage.enabled = True
        
        # Mock database operations - make second upsert fail
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_database_connector.connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.return_value = None
        
        repo_count = [0]
        def execute_side_effect(query, params=None):
            # Only fail on the INSERT/UPDATE query for the second repo
            if "INSERT INTO" in query or "ON CONFLICT" in query:
                repo_count[0] += 1
                if repo_count[0] == 2:  # Second repository's upsert fails
                    raise Exception("Database error")
        mock_cursor.execute.side_effect = execute_side_effect
        
        repositories = fetcher.fetch_trending_repositories(limit=3)
        
        with patch.object(storage, 'init_schema'):
            upserted = storage.upsert_repositories(repositories)
        
        # Should succeed for second and third repos, fail for first
        assert upserted == 2
