"""Tests for the minedd.query module."""
import os
import pytest
from minedd.query import Query


def test_load_embeddings_success():
    """Test successful loading of embeddings from a pickle file."""
    # Create query instance
    query = Query()

    # Load existing test embeddings
    test_path = os.path.join(os.path.dirname(__file__), "data/test_embeddings.pkl")

    # Verify files exists
    assert os.path.exists(test_path), f"Test embeddings pickle file not found at {test_path}"

    # Load embeddings
    result = query.load_embeddings(test_path)

    # Verify docs are loaded and method returns self
    assert query.docs is not None, "Embeddings should be loaded"
    assert result is query, "Method should return self for chaining"


def test_load_embeddings_file_not_found():
    """Test load_embeddings raises FileNotFoundError when file doesn't exist."""
    # Create query instance
    query = Query()
    nonexistent_path = "nonexistent_file.pkl"

    # Verify correct error is raised
    with pytest.raises(FileNotFoundError) as excinfo:
        query.load_embeddings(nonexistent_path)
    assert f"File {nonexistent_path} not found" in str(excinfo.value)
