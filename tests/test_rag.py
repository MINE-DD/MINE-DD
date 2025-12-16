
import unittest
from unittest.mock import MagicMock, patch, call
import numpy as np
import os
import tempfile

from minedd.rag import (
    PersistentQdrant,
    SimpleRAG,
)

class TestRag(unittest.TestCase):

    def setUp(self):
        self.mock_embeddings = MagicMock()
        self.mock_embeddings.embed_query = MagicMock(return_value=[0.1] * 768)  # Mock embedding
        self.mock_llm = MagicMock()
        self.mock_vector_store = MagicMock()

        # Mock documents
        self.mock_document1 = MagicMock()
        self.mock_document1.page_content = "test content 1"
        self.mock_document2 = MagicMock()
        self.mock_document2.page_content = "test content 2"
        self.documents = [self.mock_document1, self.mock_document2]

        # Create a temporary directory for Qdrant storage
        self.tmpdir = tempfile.mkdtemp()
        self.qdrant_path = os.path.join(self.tmpdir, "test_qdrant")

        # Patch QdrantClient for all tests
        self.qdrant_client_patcher = patch('minedd.rag.QdrantClient')
        self.mock_client_class = self.qdrant_client_patcher.start()
        self.mock_client = MagicMock()
        self.mock_client_class.return_value = self.mock_client

        # Create a shared PersistentQdrant instance
        self.persistent_qdrant = PersistentQdrant("test_collection", self.mock_embeddings, self.qdrant_path)

    def tearDown(self):
        # Clean up patches and temp directory
        self.qdrant_client_patcher.stop()
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    @patch('minedd.rag.QdrantVectorStore')
    def test_persistent_qdrant_initialize_load_existing(self, mock_vector_store_class):
        # Setup mocks for existing collection
        mock_collection = MagicMock()
        mock_collection.name = "test_collection"
        self.mock_client.get_collections.return_value.collections = [mock_collection]

        mock_vector_store = MagicMock()
        mock_vector_store.client = self.mock_client
        mock_vector_store_class.from_existing_collection.return_value = mock_vector_store

        # Initialize and verify collection was loaded
        self.persistent_qdrant.initialize()
        mock_vector_store_class.from_existing_collection.assert_called_once()

    @patch('minedd.rag.QdrantVectorStore')
    def test_persistent_qdrant_initialize_create_new(self, mock_vector_store_class):
        # Setup mocks for new collection
        self.mock_client.get_collections.return_value.collections = []  # No existing collections

        mock_vector_store = MagicMock()
        mock_vector_store_class.return_value = mock_vector_store

        mock_document = MagicMock()
        mock_document.page_content = "test content"
        documents = [mock_document]

        # Initialize and verify collection was created
        self.persistent_qdrant.initialize(documents=documents)
        self.mock_client.create_collection.assert_called_once()
        mock_vector_store.add_documents.assert_called_once()

    def test_persistent_qdrant_add_documents(self):
        # Setup mock vector store and mock collection_exists to return True
        mock_vector_store = MagicMock()
        self.persistent_qdrant.vector_store = mock_vector_store

        # Mock collection_exists to avoid trying to create a new collection
        with patch.object(self.persistent_qdrant, 'collection_exists', return_value=True):
            # Add documents and verify
            self.persistent_qdrant.add_documents(self.documents)
            mock_vector_store.add_documents.assert_called_once()

    def test_persistent_qdrant_get_all_documents(self):
        # Setup mock points returned from Qdrant
        mock_point1 = MagicMock()
        mock_point1.payload = {"page_content": "test content 1", "metadata": {}}
        mock_point2 = MagicMock()
        mock_point2.payload = {"page_content": "test content 2", "metadata": {}}
        self.mock_client.scroll.return_value = ([mock_point1, mock_point2], None)

        mock_vector_store = MagicMock()
        self.persistent_qdrant.vector_store = mock_vector_store

        # Get all documents and verify
        documents = list(self.persistent_qdrant.get_all_documents())
        self.assertEqual(len(documents), 2)
        self.assertEqual(documents[0].page_content, "test content 1")

    def test_persistent_qdrant_search(self):
        # Setup mock vector store
        mock_vector_store = MagicMock()
        mock_vector_store.similarity_search.return_value = self.documents
        self.persistent_qdrant.vector_store = mock_vector_store

        # Perform search and verify
        self.persistent_qdrant.search("query", k=2)
        mock_vector_store.similarity_search.assert_called_with("query", k=2)

    def test_simple_rag_query(self):
        rag = SimpleRAG(self.mock_embeddings, self.mock_llm, self.mock_vector_store)

        mock_result = MagicMock()
        mock_result.page_content = "some text"
        mock_result.metadata = {"parent_doc_key": "Test_Key", "pages": "1-2"}

        self.mock_vector_store.search.return_value = [mock_result]

        mock_response = MagicMock()
        mock_response.content = "This is the answer."
        self.mock_llm.invoke.return_value = mock_response

        rag.chain = MagicMock(invoke=MagicMock(return_value=mock_response))

        results, response = rag.query("What is a test?", k=1)

        self.mock_vector_store.search.assert_called_once_with("What is a test?", k=1, verbose=False)
        self.assertEqual(len(results), 1)
        self.assertEqual(response, "This is the answer.")

if __name__ == '__main__':
    unittest.main()
