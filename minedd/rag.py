# https://python.langchain.com/api_reference/community/vectorstores/langchain_community.vectorstores.faiss.FAISS.html#langchain_community.vectorstores.faiss.FAISS
# %pip install --quiet --upgrade langchain-text-splitters langchain-community langgraph 
# pip install -qU "langchain[cohere]"
# pip install -qU langchain-huggingface langchain-ollama
# pip install faiss-cpu
# pip install cohere

import re
import os
from typing import Optional
import faiss
import numpy as np
from uuid import uuid4
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction import _stop_words
import string
from tqdm import tqdm
from pathlib import Path

from langchain.retrievers.document_compressors.cross_encoder_rerank import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_ollama import OllamaEmbeddings
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document
from minedd.document import get_documents_from_directory
from dotenv import load_dotenv
load_dotenv(override=True)


CANNOT_ANSWER_PHRASE = "Sorry, I do not know the answer for this =("
CITATION_KEY_CONSTRAINTS = (
    "## Valid citation examples: \n"
    "- [Dhervs_AreHospitalizations_07a4df pages 6-6]\n"
    "- [Are_hospit pages 3-4] \n"
    "- [Global_Sea pages 7-10] \n"
    "## Invalid citation examples: \n"
    "- Example2024Example pages 3-4 and pages 4-5 \n"
    "- Example2024Example (pages 3-4) \n"
    "- Example2024Example pages 3-4, pages 5-6 \n"
    "- Example2024Example et al. (2024) \n"
    "- Example's work (pages 17–19) \n"  # noqa: RUF001
    "- (pages 17–19) \n"  # noqa: RUF001
)
QA_VANILLA_PROMPT = (
    "Answer the question below with the context.\n\n"
    "Context:\n\n{context}\n\n----\n\n"
    "Question: {question}\n\n"
    "Write an answer based on the context. "
    "If the context provides insufficient information reply "
    f'"{CANNOT_ANSWER_PHRASE}." '
    "For each part of your answer, indicate which sources support the claims you make. "
    "Each context has a citation key at the end of it, which looks like {example_citation}. "
    "Only cite from the context above and only use the citation keys from the context. "
    f"{CITATION_KEY_CONSTRAINTS}"
    "Do not concatenate citation keys, just use them as is. "
    "Write in the style of a Wikipedia article, with concise sentences and "
    "coherent paragraphs. The context comes from a variety of sources and is "
    "only a summary, so there may inaccuracies or ambiguities. If quotes are "
    "present and relevant, use them in the answer. This answer will go directly "
    "onto Wikipedia, so do not add any extraneous information.\n\n"
    "Answer ({answer_length}):"
)

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


class PersistentFAISS:
    def __init__(self, index_path, index, embeddings_engine, include_bm25=False):
        self.index_path = index_path
        self.embeddings = embeddings_engine
        # Vector index
        self.vector_index = index
        self.vector_store = None
        # BM25 index (In-memory)
        self.include_bm25 = include_bm25
        self.bm25_index = None
        # Explicit document list to retrieve from BM25 indices
        self.documents_list = []
        
    def initialize(self, documents:Optional[list[Document]]=None):
        """Initialize or load the vector store"""
        if self._index_exists():
            self._load_existing_index()
            self.documents_list = list(self.get_all_documents())
            self._load_bm25_index() if self.include_bm25 else None
        elif documents is None:
            print(f"No existing index found at {self.index_path}, creating an empty new one...") 
            self.vector_store = FAISS(
                embedding_function=self.embeddings,
                index=self.vector_index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={},
            )
        else:
            print(f"No existing index found at {self.index_path}, creating a new one with {len(documents)} documents...")    
            self._create_new_index(documents)
            if self.include_bm25:
                self.documents_list = documents
                self._load_bm25_index()
    

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
        texts = [doc.page_content for doc in self.documents_list]
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
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        os.makedirs(self.index_path, exist_ok=True)
        self.vector_store.save_local(self.index_path)
    
    def get_all_documents(self):
        """Get all documents from the vector store"""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        
        return self.vector_store.docstore._dict.values() # type: ignore

    def add_documents(self, documents):
        """Add new documents to existing index"""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        uuids = [str(uuid4()) for _ in range(len(documents))]
        self.vector_store.add_documents(documents=documents, ids=uuids)
        self._save_index()
        print(f"Added {len(documents)} documents and saved index")
    
    def keyword_and_reranking_search(self, query, top_k=3, num_candidates=10, verbose=False):
        print(f"Input question: {query}\nTop-K: {top_k}\nNum Candidates: {num_candidates}", query)
        if self.vector_store is None or self.bm25_index is None:
            raise ValueError("Vector store and BM25 index need to be initialized to use hybrid search!")

        ##### BM25 search (lexical search) #####
        bm25_scores = self.bm25_index.get_scores(bm25_tokenizer(query))
        top_n = np.argpartition(bm25_scores, -num_candidates)[-num_candidates:]
        bm25_hits = [{'corpus_id': idx, 'score': bm25_scores[idx]} for idx in top_n]
        bm25_hits = sorted(bm25_hits, key=lambda x: x['score'], reverse=True)

        if verbose:
            print("Top-K lexical search (BM25) hits")
            for hit in bm25_hits[0:top_k]:
                print("\t{:.3f}\t{}".format(hit['score'], self.documents_list[hit['corpus_id']].page_content.replace("\n", " ")))

        # Reranking with HuggingFace
        docs_for_reranking = [self.documents_list[hit['corpus_id']] for hit in bm25_hits]
        model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
        reranker = CrossEncoderReranker(model=model, top_n=top_k)
        reranked_docs = reranker.compress_documents(documents=docs_for_reranking, query=query)
        
        if verbose:
            print("Top-K reranked hits")
            for doc in reranked_docs:
                print("\t{}".format(doc.page_content.replace("\n", " ")))
        
        return reranked_docs

    def search(self, query, k, hybrid=False, num_candidates=10, verbose=False):
        """Perform similarity search"""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        
        if hybrid and self.include_bm25:
            return self.keyword_and_reranking_search(query, top_k=k, num_candidates=num_candidates, verbose=verbose)
        else:
            return self.vector_store.similarity_search(query, k=k)


class SimpleRAG:
    def __init__(self, embeddings, generative_llm, vector_store):
        self.llm = generative_llm
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.prompt = ChatPromptTemplate([("user", QA_VANILLA_PROMPT)])
        self.chain = self.prompt | self.llm

    def load_documents(self, documents):
        """Load documents into the vector store"""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        self.vector_store.add_documents(documents)
    
    def _make_context_key(self, document: Document):
        """Create a unique key for the document based on its metadata"""
        # doc_title = document.metadata.get('title', 'No Title').replace(' ', '_')[:10]
        doc_pages = document.metadata.get('pages')
        doc_key = document.metadata.get('parent_doc_key', 'NoKEY')
        full_context = f"{document.page_content} [{doc_key} pages {doc_pages}]"
        return full_context

    def _clean_context(self, context: str):
        """Remove original citation indices form the oriignal text, as it may confuse the LLM"""
        # Remove parentheses containing comma-separated numbers, e.g. (2, 3, 11, 12)
        context = re.sub(r'\(\s*\d+(?:\s*,\s*\d+)*\s*\)', '', context) 
        # Remove square brackets with numbers, e.g. [2, 3, 11, 12]
        context = re.sub(r'(?:\[\d+\])+', '', context)
        # Remove standalone numbers in square brackets, e.g. [2], [3]
        context = re.sub(r'\[\d+\]', '', context)
        # Remove standalone numbers in parentheses, e.g. (2), (3)
        context = re.sub(r'\(\d+\)[\.\,]+', '', context)
        return context

    def query(self, question, k, hybrid=False, num_candidates=10, answer_length=200, verbose=False):
        """Query the knowledge graph with natural language"""
        # Retrieve closest passages (results are a list of Documents)           
        results = self.vector_store.search(question, k=k, hybrid=hybrid, num_candidates=num_candidates, verbose=verbose)
        context = "\n\n".join([self._make_context_key(r) for r in results])
        print(f"Retrieved {len(results)} results")
        if verbose:
            print(f"Retrieved {len(results)} results for question: {question}")
            print("\t>>>",context)
        # generate Answer based on results
        response = self.chain.invoke({
            "context": context,
            "question": f"{question} Please also refer to the sources of your claims (according to the context provided to you)",
            "example_citation": "[Dhervs_AreHospitalizations_07a4df pages 6-6]", 
            "answer_length": answer_length
        })
        return results, response.content


def run_vanilla_rag(embeddings, llm):

    PAPERS_DIRECTORY = Path.home() / "papers_minedd_mini"
    SAVE_VECTOR_PATH = "minedd_test_index"
    
    # Index and Store Embeddings
    index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))
    vector_store_path = SAVE_VECTOR_PATH
    vector_store = PersistentFAISS(
        index_path=vector_store_path,
        index=index, 
        embeddings_engine=embeddings,
        include_bm25=True
        )
    
    # Create RAG Engine
    rag_engine = SimpleRAG(
        embeddings=embeddings,
        generative_llm=llm,
        vector_store=vector_store
    )

    # ## Load Docs + Create a Document object for each chunk
    if os.path.exists(vector_store_path) and os.listdir(vector_store_path):
        print(f"Loading existing vector store from {vector_store_path}")
        vector_store.initialize()
    else:
        print(f"No existing vector store found at {vector_store_path}, Chunking and Loading Documents to create one...")
        docs = get_documents_from_directory(
            directory=PAPERS_DIRECTORY,
            extensions=['.json'],
            chunk_size=10, # Number of sentences to merge into one Document
            overlap=2 # Number of sentences to overlap between chunks
        )
        vector_store.initialize(documents=docs)

    # 0 - Query Vector Store
    query = "How is campylobacter related to seasonality?"

    ## 1- Retrieval Options
    # We'll use embedding search. But ideally we'd do hybrid
    print("\n\n----- FAISS Index Simple Vector Search\n\n")
    results = vector_store.search(query, k=5, hybrid=False, num_candidates=30, verbose=True)

    for r in results:
        print(r)
    
    print("\n\n----- FAISS Index Hybrid Vector Search\n\n")
    results = vector_store.search(query, k=5, hybrid=True, num_candidates=30, verbose=True)

    for r in results:
        print(r)
    
    ## 2 - Retrieval + Generation
    print("\n\n----- RAG Response \n\n")
    contexts, response = rag_engine.query(question=query, k=5, hybrid=True, num_candidates=20, verbose=True)
    print("\n\n",response)


if __name__ == "__main__":
    # model_name="command-r-plus",
    # model_provider="cohere"
    ### OR
    # model_name="gemini-2.5-flash-lite-preview-06-17",
    # model_provider="google_genai"
    ### OR
    model_name="llama3.2:latest"
    model_provider="ollama"


    run_vanilla_rag(
        embeddings=OllamaEmbeddings(model="mxbai-embed-large:latest"),
        llm=init_chat_model(model_name, model_provider=model_provider)
    )