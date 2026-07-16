"""Unit tests for trending/logger.py module."""

import logging
import os
import tempfile
import pytest

from trending.logger import setup_logger, get_logger


def close_handlers(logger: logging.Logger) -> None:
    """Close logger-owned resources so temporary files can be removed on Windows."""
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)


@pytest.mark.unit
class TestSetupLogger:
    """Test setup_logger function."""

    def test_setup_logger_creates_logger(self):
        """Test that setup_logger creates a logger instance."""
        logger = setup_logger("test_logger")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_logger"

    def test_setup_logger_with_default_level(self):
        """Test setup_logger with default log level."""
        logger = setup_logger("test_logger_default")
        assert logger.level == logging.INFO

    def test_setup_logger_with_custom_level(self):
        """Test setup_logger with custom log level."""
        logger = setup_logger("test_logger_custom", level="DEBUG")
        assert logger.level == logging.DEBUG

    def test_setup_logger_with_file_output(self):
        """Test setup_logger with file output."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as f:
            log_file = f.name

        try:
            logger = setup_logger("test_logger_file", log_file=log_file)
            assert logger.name == "test_logger_file"
            # Check that file handler was added
            assert any(isinstance(h, logging.FileHandler) for h in logger.handlers)
        finally:
            close_handlers(logging.getLogger("test_logger_file"))
            os.unlink(log_file)

    def test_setup_logger_without_file_output(self):
        """Test setup_logger without file output (console only)."""
        logger = setup_logger("test_logger_console")
        # Should have console handler but no file handler
        assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)
        assert not any(isinstance(h, logging.FileHandler) for h in logger.handlers)

    def test_setup_logger_custom_format(self):
        """Test setup_logger with custom log format."""
        custom_format = "%(levelname)s - %(message)s"
        logger = setup_logger("test_logger_format", log_format=custom_format)
        # Check that handlers have the custom format
        for handler in logger.handlers:
            assert handler.formatter._fmt == custom_format

    def test_setup_logger_idempotent(self):
        """Test that calling setup_logger multiple times doesn't duplicate handlers."""
        logger_name = "test_logger_idempotent"
        logger1 = setup_logger(logger_name)
        initial_handler_count = len(logger1.handlers)

        logger2 = setup_logger(logger_name)
        # Should return same logger without adding more handlers
        assert logger2 is logger1
        assert len(logger2.handlers) == initial_handler_count

    def test_setup_logger_invalid_file_path(self):
        """Test setup_logger with invalid file path (should log warning)."""
        # Use an invalid path that should fail
        logger = setup_logger("test_logger_invalid", log_file="/invalid/path/that/does/not/exist.log")
        # Should still create logger, just without file handler
        assert isinstance(logger, logging.Logger)
        # Should not have file handler due to invalid path
        assert not any(isinstance(h, logging.FileHandler) for h in logger.handlers)


@pytest.mark.unit
class TestGetLogger:
    """Test get_logger function."""

    def test_get_logger_creates_new_logger(self):
        """Test that get_logger creates a new logger if it doesn't exist."""
        logger = get_logger("test_get_logger_new")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_get_logger_new"

    def test_get_logger_returns_existing_logger(self):
        """Test that get_logger returns existing logger."""
        logger_name = "test_get_logger_existing"
        logger1 = get_logger(logger_name)
        logger2 = get_logger(logger_name)
        assert logger1 is logger2

    def test_get_logger_uses_default_config(self):
        """Test that get_logger uses default configuration."""
        logger = get_logger("test_get_logger_default")
        # Should have handlers from default configuration
        assert len(logger.handlers) > 0

    def test_get_logger_after_setup_logger(self):
        """Test that get_logger returns logger created by setup_logger."""
        logger_name = "test_get_logger_after_setup"
        setup_logger(logger_name, level="DEBUG")
        logger = get_logger(logger_name)
        assert logger.level == logging.DEBUG


@pytest.mark.unit
class TestLoggerFunctionality:
    """Test actual logging functionality."""

    def test_logger_can_log_messages(self, caplog):
        """Test that logger can actually log messages."""
        logger = setup_logger("test_log_messages", level="DEBUG")
        with caplog.at_level(logging.DEBUG):
            logger.debug("Debug message")
            logger.info("Info message")
            logger.warning("Warning message")
            logger.error("Error message")

        assert "Debug message" in caplog.text
        assert "Info message" in caplog.text
        assert "Warning message" in caplog.text
        assert "Error message" in caplog.text

    def test_logger_respects_log_level(self, caplog):
        """Test that logger respects configured log level."""
        logger = setup_logger("test_log_level", level="WARNING")
        with caplog.at_level(logging.WARNING):
            logger.debug("Debug message")
            logger.info("Info message")
            logger.warning("Warning message")
            logger.error("Error message")

        # DEBUG and INFO should not appear
        assert "Debug message" not in caplog.text
        assert "Info message" not in caplog.text
        # WARNING and ERROR should appear
        assert "Warning message" in caplog.text
        assert "Error message" in caplog.text

    def test_logger_file_output(self):
        """Test that logger writes to file when configured."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as f:
            log_file = f.name

        try:
            logger = setup_logger("test_file_output", log_file=log_file)
            logger.info("Test message to file")

            # Read the file and check for message
            with open(log_file, 'r') as f:
                content = f.read()
                assert "Test message to file" in content
        finally:
            close_handlers(logging.getLogger("test_file_output"))
            os.unlink(log_file)
