"""Tests for the minedd.embeddings module."""
import pytest
import shutil
from unittest.mock import MagicMock, patch
from minedd.embeddings import Embeddings, DocumentChunk


@pytest.fixture()
def embeddings(tmp_path):
    """Fixture to create an Embeddings instance with a temporary directory."""
    temp_dir = tmp_path / "test_embeddings"
    temp_dir.mkdir()
    embeddings = Embeddings(output_embeddings_path=temp_dir/"temp_embeddings.pkl")
    yield embeddings

    # Cleanup: Remove files and directories created during the test
    shutil.rmtree(temp_dir, ignore_errors=True)

def test_print_embeddings(embeddings):
    print(embeddings)
    embeddings.load_existing_embeddings("tests/data/test_embeddings.pkl")
    print(embeddings)

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
    ordered_files = embeddings.prepare_papers("tests/mock_papers")
    details = embeddings.get_docs_details(verbose=False)
    assert details is not None


@pytest.fixture
def setup_instance():
    """Setup a test instance with mock data."""
    instance = Embeddings()
    
    # Create mock docs with sample data
    mock_doc = MagicMock()
    mock_doc.docnames = ["doc1", "doc2"]
    
    # Create mock text chunks
    mock_chunk1 = MagicMock()
    mock_chunk1.doc.docname = "doc1"
    mock_chunk1.doc.dockey = "key1"
    mock_chunk1.name = "chunk1 1-1"
    mock_chunk1.text = "This is chunk 1, Page 1 from Document 1"
    mock_chunk1.embedding = [0.1, 0.2, 0.3]
    
    mock_chunk2 = MagicMock()
    mock_chunk2.doc.docname = "doc1"
    mock_chunk2.doc.dockey = "key1"
    mock_chunk2.name = "chunk2 1-2"
    mock_chunk2.text = "This is chunk 2, Page 1 and 2 (the chunk is across pages) from Document 1"
    mock_chunk2.embedding = [0.4, 0.5, 0.6]
    
    mock_chunk3 = MagicMock()
    mock_chunk3.doc.docname = "doc2"
    mock_chunk3.doc.dockey = "key2"
    mock_chunk3.name = "chunk3 1-1"
    mock_chunk3.text = "This is chunk 1 from Document 2"
    mock_chunk3.embedding = [0.7, 0.8, 0.9]
    
    mock_doc.texts = [mock_chunk1, mock_chunk2, mock_chunk3]
    
    instance.docs = mock_doc
    return instance


def test_no_existing_embeddings(setup_instance):
    """Test when no embeddings exist."""
    instance = setup_instance
    instance.docs = None
    result = instance.get_document_chunks("doc1")
    assert result == []

    
def test_document_not_found(setup_instance):
    """Test when document is not found."""
    instance = setup_instance
    result = instance.get_document_chunks("non_existent_doc")
    assert result == []
    
def test_get_all_chunks_without_embeddings(setup_instance):
    """Test retrieving all chunks without embeddings."""
    instance = setup_instance
    result = instance.get_document_chunks("doc1")
    assert len(result) == 2
    assert all(chunk.embedding is None for chunk in result)
    assert all(chunk.docname == "doc1" for chunk in result)
    
def test_get_all_chunks_with_embeddings(setup_instance):
    """Test retrieving all chunks with embeddings."""
    instance = setup_instance
    result = instance.get_document_chunks("doc1", include_embeddings=True)
    assert len(result) == 2
    assert all(chunk.embedding is not None for chunk in result)
    
def test_get_specific_pages(setup_instance):
    """Test retrieving chunks from specific pages."""
    instance = setup_instance
    # Should only return the only chunk corresponding to page 2
    result = instance.get_document_chunks("doc1", pages=[2])
    assert len(result) == 1
    assert result[0].chunkname == "chunk2 1-2"
    
def test_get_multiple_pages(setup_instance):
    """Test retrieving chunks from multiple pages."""
    instance = setup_instance
    # Should return both chunks
    result = instance.get_document_chunks("doc1", pages=[1, 2])
    assert len(result) == 2
        
def test_get_non_existent_pages(setup_instance):
    """Test retrieving chunks from pages that don't exist."""
    instance = setup_instance
    result = instance.get_document_chunks("doc1", pages=[99])
    assert len(result) == 0
