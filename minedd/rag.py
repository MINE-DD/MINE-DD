# Qdrant-based RAG implementation with hybrid search (vector + keyword)

import re
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
    def __init__(self, embeddings_engine, qdrant_url, use_hybrid=False):
        self.embeddings_engine = embeddings_engine
        self.qdrant_url = qdrant_url
        self.is_remote = qdrant_url.startswith("http://") or qdrant_url.startswith("https://")

        # Initialize client connecting to Docker or using a local file to store the vectors
        self.client = QdrantClient(url=self.qdrant_url) if self.is_remote else QdrantClient(path=self.qdrant_url)

        # Initialize Sparse Embeddings Engine for hybrid search if needed
        self.use_hybrid = use_hybrid
        self.sparse_embeddings_engine = FastEmbedSparse(model_name="Qdrant/bm42") if self.use_hybrid else None

        # Infer vector size by embedding a test string
        test_embedding = self.embeddings_engine.embed_query("test")
        self.dense_vector_size = len(test_embedding)
        print(f"Using {self.embeddings_engine} embedder. Dense vector size: {self.dense_vector_size}")

    def init_from_filepath(self, collection_name: str, filepath: str, chunk_size: int = 10, overlap: int = 2, replace_existing: bool = False, truncate_long_docs_limit: int = 0):
        """Initialize or load collection from a filepath containing valid JSON files"""

        if replace_existing:
            self.reset_collection(collection_name)

        try:
            vector_store = self.load_existing_collection(collection_name)
        except ValueError:
            documents = get_documents_from_directory(
                directory=filepath,
                extensions=['.json'],
                chunk_size=chunk_size,  # Number of sentences to merge into one Document
                overlap=overlap  # Number of sentences to overlap between chunks
            )
            print(f"No existing collection found with name '{collection_name}', creating a new one with {len(documents)} documents...")
            if truncate_long_docs_limit > 0:
                documents = self._truncate_long_documents(documents, max_length=truncate_long_docs_limit)
            vector_store = self.create_new_collection(collection_name, documents)
        
        return vector_store

    def init_from_document_list(self, collection_name: str, documents: Optional[List[Document]] = None, replace_existing: bool = False, truncate_long_docs_limit: int = 0):
        """Initialize or load collection with a list of Document objects"""
        documents = documents if documents is not None else []

        if replace_existing:
            self.reset_collection(collection_name)

        try:
            vector_store = self.load_existing_collection(collection_name)
        except ValueError:
            print(f"No existing collection found with name '{collection_name}', creating a new one with {len(documents)} documents...")
            if truncate_long_docs_limit > 0:
                documents = self._truncate_long_documents(documents, max_length=truncate_long_docs_limit)
            vector_store = self.create_new_collection(collection_name, documents)
        
        return vector_store

    def get_collection_names(self) -> List[str]:
        """Get list of existing Qdrant collection names"""
        if self.client is None:
            raise ValueError("Qdrant client is not initialized.")
        try:
            collections = self.client.get_collections().collections
            return [col.name for col in collections]
        except Exception as e:
            raise ValueError(f"Error retrieving collection names: {e}")

    def collection_exists(self, collection_name: str) -> bool:
        """Check if Qdrant collection exists"""
        if self.client is None:
            raise ValueError("Qdrant client is not initialized.")
        try:
            collections = self.client.get_collections().collections
            return any(col.name == collection_name for col in collections)
        except Exception as e:
            raise ValueError(f"Error checking collection existence: {e}")

    def reset_collection(self, collection_name: str) -> bool:
        """Delete Qdrant collection"""
        if self.client is None:
            raise ValueError("Qdrant client is not initialized.")
        if self.collection_exists(collection_name):
            print(f"Deleting existing Qdrant collection '{collection_name}'...")
            self.client.delete_collection(collection_name)
            print("Collection deleted successfully")
            return True
        else:
            print(f"Collection '{collection_name}' does not exist, nothing to delete.")
            return False

    def load_existing_collection(self, collection_name: str) -> QdrantVectorStore:
        """Load existing Qdrant collection"""

        if not self.collection_exists(collection_name):
            raise ValueError(f"Collection '{collection_name}' does not exist.")

        if not self.is_remote:
            self.client = None

        print(f"Loading existing Qdrant collection '{collection_name}'")
        vs_params = {
            "embedding": self.embeddings_engine,
            "collection_name": collection_name,
            "vector_name": "dense"
        }

        if "http" in self.qdrant_url:
            vs_params["url"] = self.qdrant_url
        else:
            vs_params["path"] = self.qdrant_url

        # TODO: Missing the sparse vector config for hybrid
        vector_store = QdrantVectorStore.from_existing_collection(**vs_params)
        self.client = vector_store.client
        print("Collection loaded successfully")

        return vector_store

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

    def create_new_collection(self, collection_name: str, documents: list[Document]) -> QdrantVectorStore:
        """Create new Qdrant collection from documents"""
        if self.client is None:
            raise ValueError("Qdrant client is not initialized.")

        print(f"Creating new Qdrant collection '{collection_name}'...")

        if self.use_hybrid:
            # Create collection with BOTH dense and sparse vectors
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "dense": VectorParams(
                        size=self.dense_vector_size,  # Inferred from embeddings engine
                        distance=Distance.COSINE,
                        on_disk=True  # Store dense vectors on disk
                    )
                },
                sparse_vectors_config={
                    "bm42": SparseVectorParams(
                        index=models.SparseIndexParams(
                            on_disk=True  # Store BM25 on disk
                        )
                    )
                }
            )
            # Create vector store with hybrid mode
            vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=collection_name,
                embedding=self.embeddings_engine,
                sparse_embedding=self.sparse_embeddings_engine,
                retrieval_mode=RetrievalMode.HYBRID,  # Use both!
                vector_name="dense",
                sparse_vector_name="bm42"
            )
        else:
            # Create collection with only dense vectors
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "dense": VectorParams(
                        size=self.dense_vector_size,  # Inferred from embeddings engine
                        distance=Distance.COSINE,
                        on_disk=True  # Store dense vectors on disk
                    )
                }
            )
            # Create vector store with hybrid mode
            vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=collection_name,
                embedding=self.embeddings_engine,
                vector_name="dense",
            )

        # Add documents
        if len(documents) > 0:
            uuids = [str(uuid4()) for _ in range(len(documents))]
            vector_store.add_documents(documents, ids=uuids)

        print(f"New collection '{collection_name}' created and saved with {len(documents)} documents")

        return vector_store

    def get_all_documents(self, collection_name: str):
        """Get all documents from the vector store"""
        if self.client is None:
            raise ValueError("Qdrant client is not initialized.")
        
        if not self.collection_exists(collection_name):
            raise ValueError(f"Collection '{collection_name}' does not exist.")

        # Retrieve all points from Qdrant
        scroll_result = self.client.scroll(
            collection_name=collection_name,
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

    def add_documents_to_vector_store(self, vector_store: QdrantVectorStore, documents: list[Document]):
        """Add new documents to existing Vector Store"""
        
        if len(documents) > 0:
            uuids = [str(uuid4()) for _ in range(len(documents))]
            vector_store.add_documents(documents=documents, ids=uuids)
            print(f"Added {len(documents)} documents to existing collection")
            return True
        else:
            print("No documents were added")
            return False

    def search(self, vector_store, query, k, verbose=False):
        """Perform similarity search"""
        results = []
        try:
            results = vector_store.similarity_search(query, k=k)
        except Exception as e:
            print(f"Error during similarity search: {e}")
        
        if verbose:
            for res in results:
                print(f"* {res.page_content} [{res.metadata}]")
        
        return results


class SimpleRAG:
    def __init__(self, embeddings_engine, generative_llm, vector_db: PersistentQdrant):
        self.llm = generative_llm
        self.embeddings_engine = embeddings_engine
        self.vector_db = vector_db
        self.prompt = ChatPromptTemplate([("user", QA_VANILLA_PROMPT)])
        self.chain = self.prompt | self.llm
    
    def _make_context_key(self, document: Document):
        """Create a unique key for the document based on its metadata"""
        # doc_title = document.metadata.get('title', 'No Title').replace(' ', '_')[:10]
        doc_pages = document.metadata.get('pages')
        doc_key = document.metadata.get('parent_doc_key', 'NoKEY')
        full_context = f"{self._clean_context(document.page_content)} [{doc_key} pages {doc_pages}]"
        return full_context

    def _clean_context(self, context: str):
        """Remove original citation indices form the oriignal text, as it may confuse the LLM"""
        # Remove parentheses containing comma-separated numbers, e.g. (2, 3, 11, 12)
        context = re.sub(r'\(\s*\d+(?:\s*,\s*\d+)*\s*\)', '', context)
        # Remove square brackets containing comma-separated numbers, e.g. [2, 3, 11, 12] or [7,32]
        context = re.sub(r'\[\s*\d+(?:\s*,\s*\d+)*\s*\]', '', context)
        # Remove square brackets with numbers, e.g. [2, 3, 11, 12]
        context = re.sub(r'(?:\[\d+\])+', '', context)
        # Remove standalone numbers in square brackets, e.g. [2], [3]
        context = re.sub(r'\[\d+\]', '', context)
        # Remove standalone numbers in parentheses, e.g. (2), (3)
        context = re.sub(r'\(\d+\)[\.\,]+', '', context)
        return context

    def query(self, vector_store, question, k, answer_length=200, verbose=False):
        # Retrieve closest passages (results are a list of LangChain Documents)           
        results = self.vector_db.search(vector_store, question, k=k, verbose=verbose)
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
    QDRANT_URL = "http://localhost:6333" #OR local_file.db OR #os.getenv("QDRANT_URL", "http://localhost:6333")

    # Initialize Qdrant Vector Store
    qdrant_db = PersistentQdrant(
        embeddings_engine=embeddings_engine,
        qdrant_url=QDRANT_URL,
        use_hybrid=False  # Set to True to enable hybrid search (dense + sparse)
    )

    # Load Docs + Create a Document object for each chunk
    vector_store_1 = qdrant_db.init_from_filepath(
        collection_name=COLLECTION_NAME,
        filepath=PAPERS_DIRECTORY,
        chunk_size=10,  # Number of sentences to merge into one Document
        overlap=2,      # Number of sentences to overlap between chunks
        replace_existing=False
    )

    vector_store_2 = qdrant_db.init_from_filepath(
        collection_name="minedd_rag_mini_2",
        filepath=PAPERS_DIRECTORY,
        chunk_size=20,  # Number of sentences to merge into one Document
        overlap=10,      # Number of sentences to overlap between chunks
        truncate_long_docs_limit=1000,  # Truncate long documents to 1000 characters    
        replace_existing=False
    )
    
    # Create RAG Engine
    rag_engine = SimpleRAG(
        embeddings_engine=embeddings_engine,
        generative_llm=llm,
        vector_db=qdrant_db
    )

    # Print available collections
    print("\nAvailable Qdrant Collections:")
    collections = qdrant_db.get_collection_names()
    for col in collections:
        print(f"* {col}")

    # Query Vector Store
    query = "How is campylobacter related to seasonality?"

    ## Retrieval + Generation
    print("\n\n========== RAG Response 1 ===========\n\n")
    contexts, final_response = rag_engine.query(vector_store_1, question=query, k=5, verbose=False)
    print([len(c.page_content) for c in contexts])
    print("\n\n******* FINAL RESPONSE:\n", final_response)

    print("\n\n========== RAG Response 2 ===========\n\n")
    contexts, final_response = rag_engine.query(vector_store_2, question=query, k=5, verbose=False)
    print([len(c.page_content) for c in contexts])
    print("\n\n******* FINAL RESPONSE:\n", final_response)


if __name__ == "__main__":
    # Just testing the RAG with the "default" Embedder + Ollama model which is Llama3.2:3B
    run_vanilla_rag(
        embeddings_engine=OllamaEmbeddings(model="mxbai-embed-large:latest"),
        llm=init_chat_model("llama3.2:latest", model_provider="ollama")
    )