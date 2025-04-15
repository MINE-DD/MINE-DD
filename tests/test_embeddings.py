"""Tests for the minedd.embeddings module."""
import pytest
# import os
# import json
import shutil
from minedd.embeddings import Embeddings

@pytest.fixture()
def embeddings(tmp_path):
    """Fixture to create an Embeddings instance with a temporary directory."""
    temp_dir = tmp_path / "test_embeddings"
    temp_dir.mkdir()
    embeddings = Embeddings(output_embeddings_path=temp_dir/"temp_embeddings.pkl")
    yield embeddings

    # Cleanup: Remove files and directories created during the test
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_load_existing_embeddings(embeddings):
    """Test the load_existing_embeddings method."""
    embeddings.load_existing_embeddings("tests/data/test_embeddings.pkl")
    assert embeddings.docs is not None
    assert len(embeddings.docs.docs) == 1

def test_prepare_papers(embeddings):
    """Test the prepare_papers method."""
    ordered_files = embeddings.prepare_papers("tests/mock_papers")
    assert ordered_files == ["nihms-620915.pdf"]

def test_get_docs_details(embeddings):
    """Test the get_docs_details method."""
    embeddings.load_existing_embeddings("tests/data/test_embeddings.pkl")
    details = embeddings.get_docs_details(verbose=False)
    assert details is not None
    assert len(details) == 1