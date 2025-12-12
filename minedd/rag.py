# Qdrant-based RAG implementation with hybrid search (vector + keyword)

import re
import os
from typing import Optional, List
from uuid import uuid4

from langchain_ollama import OllamaEmbeddings
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, SparseVectorParams, VectorParams
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse

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

class PersistentQdrant:
    def __init__(self, collection_name, embeddings_engine, qdrant_url, use_hybrid=False):
        self.collection_name = collection_name
        self.embeddings_engine = embeddings_engine
        self.sparse_embeddings_engine = FastEmbedSparse(model_name="Qdrant/bm25") if use_hybrid else None
        self.qdrant_url = qdrant_url
        self.use_hybrid = use_hybrid

        # Initialize Qdrant client
        self.client = QdrantClient(path=qdrant_url) if qdrant_url.startswith("file://") else QdrantClient(url=qdrant_url)
        self.vector_store = None

        # Infer vector size by embedding a test string
        test_embedding = self.embeddings_engine.embed_query("test")
        self.dense_vector_size = len(test_embedding)
        print(f"Using {self.embeddings_engine} embedder. Dense vector size: {self.dense_vector_size}")

    def initialize(self, documents: Optional[List[Document]] = None, truncate_long_docs_limit=0):
        """Initialize or load the vector store"""
        if self._collection_exists():
            self._load_existing_collection()
        elif documents is None:
            print(f"No existing collection found with name '{self.collection_name}', please provide documents to create a new one...")
        elif len(documents) == 0:
            raise ValueError(f"No valid documents were provided to initialize the collection! ({len(documents)} documents)")
        else:
            print(f"No existing collection found with name '{self.collection_name}', creating a new one with {len(documents)} documents...")
            if truncate_long_docs_limit > 0:
                documents = self._truncate_long_documents(documents, max_length=truncate_long_docs_limit)
            self._create_new_collection(documents)

    def _collection_exists(self):
        """Check if Qdrant collection exists"""
        try:
            collections = self.client.get_collections().collections
            return any(col.name == self.collection_name for col in collections)
        except Exception as e:
            print(f"Error checking collection existence: {e}")
            return False

    def delete_collection(self, collection_name: str):
        """Delete Qdrant collection"""
        if self._collection_exists():
            print(f"Deleting existing Qdrant collection '{collection_name}'...")
            self.client.delete_collection(collection_name)
            print("Collection deleted successfully")
        else:
            print(f"Collection '{collection_name}' does not exist, nothing to delete.")

    def _load_existing_collection(self):
        """Load existing Qdrant collection"""
        print(f"Loading existing Qdrant collection '{self.collection_name}'")
        self.vector_store = QdrantVectorStore.from_existing_collection(
                embedding=self.embeddings_engine,
                collection_name=self.collection_name,
                url=self.qdrant_url,
            )
        print("Collection loaded successfully")

    def _truncate_long_documents(self, documents, max_length):
        """Truncate documents that exceed max_length"""
        truncated_docs = []
        for doc in documents:
            if len(doc.page_content) > max_length:
                truncated_content = doc.page_content[:max_length]
                truncated_doc = Document(
                    page_content=truncated_content,
                    metadata=doc.metadata
                )
                truncated_docs.append(truncated_doc)
            else:
                truncated_docs.append(doc)
        return truncated_docs

    def _create_new_collection(self, documents):
        """Create new Qdrant collection from documents"""
        print(f"Creating new Qdrant collection '{self.collection_name}'...")

        if self.use_hybrid:
            # Create collection with BOTH dense and sparse vectors
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": VectorParams(
                        size=self.dense_vector_size,  # Inferred from embeddings engine
                        distance=Distance.COSINE,
                        on_disk=True  # Store dense vectors on disk
                    )
                },
                sparse_vectors_config={
                    "bm25": SparseVectorParams(
                        index=models.SparseIndexParams(
                            on_disk=True  # Store BM25 on disk
                        )
                    )
                }
            )
            # Create vector store with hybrid mode
            self.vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                embedding=self.embeddings_engine,
                sparse_embedding=self.sparse_embeddings_engine,
                retrieval_mode=RetrievalMode.HYBRID,  # Use both!
                vector_name="dense",
                sparse_vector_name="bm25"
            )
        else:
            # Create collection with only dense vectors
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": VectorParams(
                        size=self.dense_vector_size,  # Inferred from embeddings engine
                        distance=Distance.COSINE,
                        on_disk=True  # Store dense vectors on disk
                    )
                }
            )
            # Create vector store with hybrid mode
            self.vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                embedding=self.embeddings_engine,
                vector_name="dense",
            )

        # Add documents
        uuids = [str(uuid4()) for _ in range(len(documents))]
        self.vector_store.add_documents(documents, ids=uuids)

        print(f"New collection '{self.collection_name}' created and saved with {len(documents)} documents")

    def get_all_documents(self):
        """Get all documents from the vector store"""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")

        # Retrieve all points from Qdrant
        scroll_result = self.client.scroll(
            collection_name=self.collection_name,
            limit=10000,  # Adjust based on your needs
            with_payload=True,
            with_vectors=False,
        )

        for point in scroll_result[0]:
            if point.payload is None:
                print(f"Warning: Point ID {point.id} has no payload, skipping...")
                continue
            doc = Document(
                page_content=point.payload.get("page_content", ""),
                metadata=point.payload.get("metadata", {})
            )
            yield doc

    def add_documents(self, documents):
        """Add new documents to existing collection"""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")

        uuids = [str(uuid4()) for _ in range(len(documents))]
        self.vector_store.add_documents(documents=documents, ids=uuids)
        print(f"Added {len(documents)} documents to collection")

    def search(self, query, k, verbose=False):
        """Perform similarity search"""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        
        results = self.vector_store.similarity_search(query, k=k)
        
        if verbose:
            for res in results:
                print(f"* {res.page_content} [{res.metadata}]")
        
        return results


class SimpleRAG:
    def __init__(self, embeddings_engine, generative_llm, vector_store):
        self.llm = generative_llm
        self.embeddings_engine = embeddings_engine
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

    def query(self, question, k, answer_length=200, verbose=False):
        # Retrieve closest passages (results are a list of LangChain Documents)           
        results = self.vector_store.search(question, k=k, verbose=verbose)
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


def run_vanilla_rag(embeddings_engine, llm):

    PAPERS_DIRECTORY = "/Users/jose/papers_minedd_mini" #os.getenv("PAPERS_DIRECTORY", "papers_minedd")
    COLLECTION_NAME = "minedd_rag_mini" #os.getenv("QDRANT_COLLECTION_NAME", "minedd_rag_collection")
    QDRANT_URL = "http://localhost:6333" #os.getenv("QDRANT_URL", "http://localhost:6333")

    # Initialize Qdrant Vector Store
    qdrant_db = PersistentQdrant(
        collection_name=COLLECTION_NAME,
        embeddings_engine=embeddings_engine,
        qdrant_url=QDRANT_URL
    )


    # Create RAG Engine
    rag_engine = SimpleRAG(
        embeddings_engine=embeddings_engine,
        generative_llm=llm,
        vector_store=qdrant_db
    )

    qdrant_db.delete_collection(collection_name="minedd_rag_mini")  # For testing purposes, delete existing collection
    # Load Docs + Create a Document object for each chunk
    if qdrant_db._collection_exists():
        docs = []
    else:
        print("No existing collection found, chunking and loading documents to create one...")
        docs = get_documents_from_directory(
            directory=PAPERS_DIRECTORY,
            extensions=['.json'],
            chunk_size=10,  # Number of sentences to merge into one Document
            overlap=2  # Number of sentences to overlap between chunks
        )
    qdrant_db.initialize(documents=docs)

    # Query Vector Store
    query = "How is campylobacter related to seasonality?"

    ## 1- Retrieval Options
    print("\n\n----- Qdrant Simple Vector Search\n\n")
    results = qdrant_db.search(query, k=5, verbose=True)

    for r in results:
        print(r)

    ## 2 - Retrieval + Generation
    print("\n\n========== RAG Response ===========\n\n")
    contexts, response = rag_engine.query(question=query, k=5, verbose=True)
    print("\n\n", response)


if __name__ == "__main__":
    # Just testing the RAG with the "default" Embedder + Ollama model which is Llama3.2:3B
    run_vanilla_rag(
        embeddings_engine=OllamaEmbeddings(model="mxbai-embed-large:latest"),
        llm=init_chat_model("llama3.2:latest", model_provider="ollama")
    )