
import unittest
from unittest.mock import MagicMock, patch, call
import numpy as np
import os

from minedd.rag import (
    bm25_tokenizer,
    get_bm25_index,
    PersistentFAISS,
    SimpleRAG,
)

class TestRag(unittest.TestCase):

    def test_bm25_tokenizer(self):
        text = "This is a test sentence for the tokenizer."
        expected_tokens = ["test", "sentence", "tokenizer"]
        self.assertEqual(bm25_tokenizer(text), expected_tokens)

    def test_get_bm25_index(self):
        texts = ["This is a test.", "This is another test."]
        bm25 = get_bm25_index(texts)
        self.assertIsNotNone(bm25)


    def setUp(self):
        self.mock_embeddings = MagicMock()
        self.mock_llm = MagicMock()
        self.mock_vector_store = MagicMock()
        self.mock_index = MagicMock()
        
        # Mock documents
        self.mock_document1 = MagicMock()
        self.mock_document1.page_content = "test content 1"
        self.mock_document2 = MagicMock()
        self.mock_document2.page_content = "test content 2"
        self.documents = [self.mock_document1, self.mock_document2]
        
        # Instantiate PersistentFAISS for testing
        self.persistent_faiss = PersistentFAISS("dummy_path", self.mock_index, self.mock_embeddings)
        self.persistent_faiss.vector_store = MagicMock() # Mock vector_store for most tests

    @patch('minedd.rag.os.path.exists')
    @patch('minedd.rag.os.listdir')
    @patch('minedd.rag.FAISS')
    def test_persistent_faiss_initialize_load_existing(self, mock_faiss, mock_listdir, mock_exists):
        mock_exists.return_value = True
        mock_listdir.return_value = ['index.faiss']
        
        persistent_faiss = PersistentFAISS("dummy_path", self.mock_index, self.mock_embeddings)
        persistent_faiss.initialize()
        
        mock_faiss.load_local.assert_called_once_with("dummy_path", self.mock_embeddings, allow_dangerous_deserialization=True)

    @patch('minedd.rag.os.path.exists')
    @patch('minedd.rag.FAISS')
    def test_persistent_faiss_initialize_create_new(self, mock_faiss, mock_exists):
        mock_exists.return_value = False
        
        mock_document = MagicMock()
        mock_document.page_content = "test content"
        documents = [mock_document]
        
        persistent_faiss = PersistentFAISS("dummy_path", self.mock_index, self.mock_embeddings)
        with patch.object(persistent_faiss, '_save_index') as mock_save:
            persistent_faiss.initialize(documents=documents)
            mock_faiss.from_documents.assert_called_once_with(documents, self.mock_embeddings)
            mock_save.assert_called_once()

    def test_persistent_faiss_add_documents(self):
        with patch.object(self.persistent_faiss, '_save_index') as mock_save:
            self.persistent_faiss.add_documents(self.documents)
            self.persistent_faiss.vector_store.add_documents.assert_called_once()
            mock_save.assert_called_once()

    def test_persistent_faiss_get_all_documents(self):
        self.persistent_faiss.vector_store.docstore._dict.values.return_value = self.documents
        documents = self.persistent_faiss.get_all_documents()
        self.assertEqual(documents, self.documents)

    def test_persistent_faiss_search(self):
        self.persistent_faiss.search("query", k=2)
        self.persistent_faiss.vector_store.similarity_search.assert_called_with("query", k=2)

    def test_simple_rag_query(self):
        rag = SimpleRAG(self.mock_embeddings, self.mock_llm, self.mock_vector_store)
        
        mock_result = MagicMock()
        mock_result.page_content = "some text"
        mock_result.metadata = {"title": "Test Title", "pages": "1-2"}
        
        self.mock_vector_store.search.return_value = [mock_result]
        
        mock_response = MagicMock()
        mock_response.content = "This is the answer."
        self.mock_llm.invoke.return_value = mock_response
        
        rag.chain = MagicMock(invoke=MagicMock(return_value=mock_response))

        results, response = rag.query("What is a test?", k=1)
        
        self.mock_vector_store.search.assert_called_once_with("What is a test?", k=1, hybrid=False, num_candidates=10, verbose=False)
        self.assertEqual(len(results), 1)
        self.assertEqual(response, "This is the answer.")

if __name__ == '__main__':
    unittest.main()
