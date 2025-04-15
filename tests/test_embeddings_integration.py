"""Integration tests for minedd.query that require Ollama and local GPU.

These tests verify the full functionality of the `process_papers` method,
but require a local environment with Ollama running. These tests are skipped
in CI environments.

To run these tests locally:
1. Make sure Ollama is running (`ollama serve`)
2. Run: `pytest tests/test_embeddings_integration.py -v` or `pytest -m "integration" -v`
"""

import os
import pytest
import pandas as pd
import numpy as np
from minedd.embeddings import Embeddings

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration

# Skip these tests in CI environments
if os.environ.get("CI") == "true" or os.environ.get("GITHUB_ACTIONS") == "true":
    pytestmark = pytest.mark.skip(reason="Integration tests requiring Ollama - skipped in CI environment")

# Global flag to optionally skip even in local environments
# Set to False when you want to run the integration tests locally
SKIP_OLLAMA_TESTS = False

if SKIP_OLLAMA_TESTS:
    pytestmark = pytest.mark.skip(reason="Ollama integration tests explicitly disabled with SKIP_OLLAMA_TESTS=True")


class TestEmbeddingsIntegration:
    """Integration tests for the Query class that require Ollama."""

    @pytest.fixture(scope="class")
    def embeddings_engine(self):
        """Create a Query instance for testing."""
        engine = Embeddings(
            model="ollama/llama3.2:1b",
            embedding_model="ollama/mxbai-embed-large:latest",
            output_embeddings_path="tests/data/test_embeddings.pkl",
            paper_directory="tests/mock_papers/",
            existing_docs=None,
        )
        return engine


    def test_process_papers(self, embeddings_engine):
        """Test creating paper embeddings with the default papers directory."""
        paper_files = embeddings_engine.prepare_papers()
        # Confirm the engine found 1 valid papers in the test directory
        assert len(paper_files) == 1, "No papers found in the specified directory."
        embeddings_engine.process_papers(paper_files)
        # Verify the output file exists
        assert os.path.exists(embeddings_engine.output_embeddings_path), "Embeddings file not created."
        # Verify embeddings docs are created
        assert embeddings_engine.docs is not None, "Embeddings docs are not created."

