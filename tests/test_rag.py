
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

        # Create a shared PersistentQdrant instance (new signature)
        self.persistent_qdrant = PersistentQdrant(self.mock_embeddings, self.qdrant_path, use_hybrid=False)

    def tearDown(self):
        # Clean up patches and temp directory
        self.qdrant_client_patcher.stop()
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    @patch('minedd.rag.QdrantVectorStore')
    def test_persistent_qdrant_init_from_document_list_load_existing(self, mock_vector_store_class):
        # Setup mocks for existing collection
        mock_collection = MagicMock()
        mock_collection.name = "test_collection"
        self.mock_client.get_collections.return_value.collections = [mock_collection]

        mock_vector_store = MagicMock()
        mock_vector_store.client = self.mock_client
        mock_vector_store_class.from_existing_collection.return_value = mock_vector_store

        # Initialize and verify collection was loaded
        vector_store = self.persistent_qdrant.init_from_document_list(
            collection_name="test_collection",
            documents=self.documents, #type: ignore
            replace_existing=False
        )
        mock_vector_store_class.from_existing_collection.assert_called_once()
        self.assertIsNotNone(vector_store)

    @patch('minedd.rag.QdrantVectorStore')
    def test_persistent_qdrant_init_from_document_list_create_new(self, mock_vector_store_class):
        # Setup mocks for new collection
        self.mock_client.get_collections.return_value.collections = []  # No existing collections

        mock_vector_store = MagicMock()
        mock_vector_store_class.return_value = mock_vector_store

        mock_document = MagicMock()
        mock_document.page_content = "test content"
        documents = [mock_document]

        # Initialize and verify collection was created
        vector_store = self.persistent_qdrant.init_from_document_list(
            collection_name="test_collection",
            documents=documents, #type: ignore
            replace_existing=False
        )
        self.mock_client.create_collection.assert_called_once()
        mock_vector_store.add_documents.assert_called_once()
        self.assertIsNotNone(vector_store)

    def test_persistent_qdrant_add_documents_to_vector_store(self):
        # Setup mock vector store
        mock_vector_store = MagicMock()

        # Add documents and verify
        result = self.persistent_qdrant.add_documents_to_vector_store(mock_vector_store, self.documents) #type: ignore
        mock_vector_store.add_documents.assert_called_once()
        self.assertTrue(result)

    def test_persistent_qdrant_get_all_documents(self):
        # Setup mock collection
        mock_collection = MagicMock()
        mock_collection.name = "test_collection"
        self.mock_client.get_collections.return_value.collections = [mock_collection]

        # Setup mock points returned from Qdrant
        mock_point1 = MagicMock()
        mock_point1.payload = {"page_content": "test content 1", "metadata": {}}
        mock_point2 = MagicMock()
        mock_point2.payload = {"page_content": "test content 2", "metadata": {}}
        self.mock_client.scroll.return_value = ([mock_point1, mock_point2], None)

        # Get all documents and verify
        documents = list(self.persistent_qdrant.get_all_documents("test_collection"))
        self.assertEqual(len(documents), 2)
        self.assertEqual(documents[0].page_content, "test content 1")

    def test_persistent_qdrant_search(self):
        # Setup mock vector store
        mock_vector_store = MagicMock()
        mock_vector_store.similarity_search.return_value = self.documents

        # Perform search and verify
        results = self.persistent_qdrant.search(mock_vector_store, "query", k=2)
        mock_vector_store.similarity_search.assert_called_with("query", k=2)
        self.assertEqual(len(results), 2)

    def test_simple_rag_query(self):
        # Create mock PersistentQdrant
        mock_persistent_qdrant = MagicMock()
        mock_result = MagicMock()
        mock_result.page_content = "some text"
        mock_result.metadata = {"parent_doc_key": "Test_Key", "pages": "1-2"}
        mock_persistent_qdrant.search.return_value = [mock_result]

        # Create SimpleRAG with the correct signature
        rag = SimpleRAG(self.mock_embeddings, self.mock_llm, mock_persistent_qdrant)

        mock_response = MagicMock()
        mock_response.content = "This is the answer."
        rag.chain = MagicMock(invoke=MagicMock(return_value=mock_response))

        # Mock vector store for query
        mock_vector_store = MagicMock()

        # Execute query with vector_store as first parameter
        results, response = rag.query(mock_vector_store, "What is a test?", k=1)

        mock_persistent_qdrant.search.assert_called_once_with(mock_vector_store, "What is a test?", k=1, verbose=False)
        self.assertEqual(len(results), 1)
        self.assertEqual(response, "This is the answer.")

if __name__ == '__main__':
    unittest.main()
