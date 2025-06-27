# https://python.langchain.com/api_reference/community/vectorstores/langchain_community.vectorstores.faiss.FAISS.html#langchain_community.vectorstores.faiss.FAISS
# %pip install --quiet --upgrade langchain-text-splitters langchain-community langgraph 
# pip install -qU "langchain[cohere]"
# pip install -qU langchain-huggingface OR langchain-ollama
# pip install faiss-cpu
# pip install cohere

import os
import cohere
import faiss
import numpy as np
import pandas as pd
from uuid import uuid4
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction import _stop_words
import string
from tqdm import tqdm
from pathlib import Path
import json

from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain.chat_models import init_chat_model
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from minedd.document import DocumentMarkdown
from minedd.document import DocumentPDF
from dotenv import load_dotenv
load_dotenv(override=True)


# This function is to create a document database from a list of markdown files.
def documentdb_from_markdown(filenames):
    chunks = []
    # Load Document(s)
    for filename in filenames:
        markdown_paper = DocumentMarkdown(md_path=filename)
        chunks += markdown_paper.convert_to_chunks(mode="chars", chunk_size=1500, overlap=100)
    # Index
    print(len(chunks))
    response = co.embed(
    texts=chunks,
    input_type="search_document",
    ).embeddings

    embeds = np.array(response)
    embedding_dim = embeds.shape[1]
    print(embeds.shape)
    # Create Index (and Vector Store?)
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(np.float32(embeds))
    return index, chunks


def bm25_tokenizer(text):
    tokenized_doc = []
    for token in text.lower().split():
        token = token.strip(string.punctuation)
        if len(token) > 0 and token not in _stop_words.ENGLISH_STOP_WORDS:
            tokenized_doc.append(token)
    return tokenized_doc


def get_bm25_index(texts):
    tokenized_corpus = []
    for passage in tqdm(texts):
        tokenized_corpus.append(bm25_tokenizer(passage))
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25


def keyword_and_reranking_search(texts, bm25, query, top_k=3, num_candidates=10):
    print("Input question:", query)

    ##### BM25 search (lexical search) #####
    bm25_scores = bm25.get_scores(bm25_tokenizer(query))
    top_n = np.argpartition(bm25_scores, -num_candidates)[-num_candidates:]
    bm25_hits = [{'corpus_id': idx, 'score': bm25_scores[idx]} for idx in top_n]
    bm25_hits = sorted(bm25_hits, key=lambda x: x['score'], reverse=True)

    print(f"Top-3 lexical search (BM25) hits")
    for hit in bm25_hits[0:top_k]:
        print("\t{:.3f}\t{}".format(hit['score'], texts[hit['corpus_id']].replace("\n", " ")))

    #Add re-ranking
    docs = [texts[hit['corpus_id']] for hit in bm25_hits]

    print(f"\nTop-3 hits by rank-API ({len(bm25_hits)} BM25 hits re-ranked)")
    results = co.rerank(query=query, documents=docs, top_n=top_k, return_documents=True)
    return results.results


class PersistentFAISS:
    def __init__(self, index_path, index, embeddings_engine, include_bm25=False):
        self.index_path = index_path
        self.embeddings = embeddings_engine
        self.vector_index = index
        self.vector_store = None
        self.include_bm25 = include_bm25
        self.bm25_index = None
        
    def initialize(self, documents=None):
        """Initialize or load the vector store"""
        if self._index_exists():
            self._load_existing_index()
            self._load_bm25_index() if self.include_bm25 else None
        elif documents is None:
            print("No existing index found, creating an empty new one...") 
            self.vector_store = FAISS(
                embedding_function=self.embeddings,
                index=self.vector_index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={},
            )
        else:
            print(f"No existing index found, creating a new one with {len(documents)} documents...")    
            self._create_new_index(documents)
            self._load_bm25_index() if self.include_bm25 else None
    

    def _index_exists(self):
        """Check if FAISS index exists"""
        return os.path.exists(self.index_path) and os.listdir(self.index_path)
    
    def _load_existing_index(self):
        """Load existing FAISS index"""
        print(f"Loading existing FAISS index from {self.index_path}")
        self.vector_store = FAISS.load_local(
            self.index_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        print("Index loaded successfully")

    def _load_bm25_index(self):
        tokenized_corpus = []
        texts = [doc.page_content for doc in self.get_all_documents()]
        for passage in tqdm(texts):
            tokenized_corpus.append(bm25_tokenizer(passage))
        self.bm25_index = BM25Okapi(tokenized_corpus)

    def _create_new_index(self, documents):
        """Create new FAISS index from documents"""
        print("Creating new FAISS index...")
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        self._save_index()
        print("New index created and saved")
    
    def _save_index(self):
        """Save the current index"""
        os.makedirs(self.index_path, exist_ok=True)
        self.vector_store.save_local(self.index_path)
    
    def get_all_documents(self):
        """Get all documents from the vector store"""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        
        return self.vector_store.docstore._dict.values()

    def add_documents(self, documents):
        """Add new documents to existing index"""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        uuids = [str(uuid4()) for _ in range(len(documents))]
        self.vector_store.add_documents(documents=documents, ids=uuids)
        self._save_index()
        print(f"Added {len(documents)} documents and saved index")
    
    def search(self, query, k=3, hybrid=False, num_candidates=10):
        """Perform similarity search"""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        
        if hybrid and self.include_bm25:
            return keyword_and_reranking_search(self.bm25, query, top_k=k, num_candidates=10)
        else:
            return self.vector_store.similarity_search(query, k=k)


class SimpleRAG:
    def __init__(self, embeddings, generative_llm, vector_store):
        self.llm = generative_llm
        self.embeddings = embeddings
        self.vector_store = vector_store

    def get_documents_from_directory(self, directory, extensions=['.json'], chunk_size=10, overlap=5):
        """Pre-process a directory of documents and load them into the vector store"""
        # Load documents from directory
        documents = []
        processed_files = 0
        for filename in os.listdir(directory):
            if '.json' in extensions and filename.endswith('.json'):
                processed_files += 1
                full_paper = DocumentPDF.from_json(json_path=os.path.join(directory, filename))
                paper_chunks = full_paper.get_chunks(
                    mode='from_json', 
                    chunk_size=chunk_size, 
                    overlap=overlap, 
                    as_langchain_docs=True
                    )
                documents.extend(paper_chunks)

        print(f"Processed {processed_files} files, extracted {len(documents)} Document chunks")
        return documents

    def load_documents(self, documents):
        """Load documents into the vector store"""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        self.vector_store.add_documents(documents)


def run_vanilla_rag(embeddings, llm):
    
    # Index and Store Embeddings
    index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))
    vector_store_path = "minedd_test_index"
    vector_store = PersistentFAISS(
        index_path=vector_store_path,
        index=index, 
        embeddings_engine=embeddings,
        include_bm25=True
        )
    
    # Create RAG Engine
    rag_engine = SimpleRAG(
        embeddings=ollama_embeddings,
        generative_llm=llm,
        vector_store=vector_store
    )

    # ## Load Docs + Create a Document object for each chunk
    if os.path.exists(vector_store_path) and os.listdir(vector_store_path):
        print(f"Loading existing vector store from {vector_store_path}")
        vector_store.initialize()
    else:
        print(f"No existing vector store found at {vector_store_path}, Chunking and Loading Documents to create one...")
        docs = rag_engine.get_documents_from_directory(
            directory=Path.home() / "papers_minedd",
            extensions=['.json'],
            chunk_size=8, # Number of sentences to merge into one Document
            overlap=2 # Number of sentences to overlap between chunks
        )
        vector_store.initialize(documents=docs)

    # 0 - Query Vector Store
    query = "How is rotavirus related to seasonality? Please also refer to the sources of your claims (according to the context provided to you)"

    # 1- Retrieval
    # We'll use embedding search. But ideally we'd do hybrid
    print("\n\n----- FAISS Index Vector Search\n\n")
    results = vector_store.search(query, k=3)

    for r in results:
        print(r)


    ## 2 - ReRanking (Using Cohere?)
    print("\n\n----- BM25 + ReRanker\n\n")
    texts = [doc.page_content for doc in vector_store.get_all_documents()]
    results_hybrid = keyword_and_reranking_search(texts, vector_store.bm25_index, query, top_k=3, num_candidates=10)
    for hit in results_hybrid:
        print("\t{:.3f}\t{}".format(hit.relevance_score, hit.document.text.replace("\n", " ")))

    # 3 - Grounded Generation (Using Cohere?)
    docs_dict = [{'text': doc.page_content} for doc in results]
    response = co.chat(
        message = query,
        documents=docs_dict
    )
    print(f"Response FAISS:\n{response.text}")


    docs_dict = [{'text': doc.document.text} for doc in results_hybrid]
    response = co.chat(
        message = query,
        documents=docs_dict
    )
    print(f"Response Hybrid ReRanked:\n{response.text}")


if __name__ == "__main__":
    # model_name="command-r-plus",
    # model_provider="cohere"
    ### OR
    # model_name="gemini-2.5-flash-lite-preview-06-17",
    # model_provider="google_genai"
    ### OR
    # model_name="llama3.2:latest"
    # model_provider="ollama"

    model_name="command-r-plus"
    model_provider="cohere"

    # For now will use Cohere Re-Ranker
    cohere_api_key = os.getenv('COHERE_API_KEY')
    co = cohere.Client(cohere_api_key)

    llm = init_chat_model(model_name, model_provider=model_provider)
    # llm = init_chat_model("gemini-2.5-flash-lite-preview-06-17", model_provider="google_genai")

    # Embeddings model
    ollama_embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")

    run_vanilla_rag(
        embeddings=ollama_embeddings,
        llm=llm
    )