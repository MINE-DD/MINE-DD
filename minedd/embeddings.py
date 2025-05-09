import os
import re
import pandas as pd
from dataclasses import dataclass
from tqdm import tqdm
from paperqa import Docs
import pickle as pkl
from minedd.utils import configure_settings, safely_load_pickle_file

@dataclass
class DocumentChunk:
    """Class to represent a chunk of a document.
    This class is used to store the text and embedding of a chunk of a document.
    It also contains metadata with provenance information.

    Attributes:
        text (str): The text of the chunk.
        docname (str): The name of the document the chunk belongs to.
        dockey (str): The key of the document the chunk belongs to.
        chunkname (str): The name of the chunk.
        pages (list[int]): The page numbers of the chunk.
        embedding (list[float], optional): The embedding of the chunk. Defaults to None.
    """
    text: str
    docname: str
    dockey: str
    chunkname: str
    pages: list[int]
    embedding: list[float] = None
    
    def __repr__(self):
        embedding = self.embedding[:5] if self.embedding is not None else []
        return f"DocumentChunk(text={self.text[:30]}, embedding={embedding}..., docname={self.docname}, dockey={self.dockey}, chunkname={self.chunkname}, pages={self.pages})"


class Embeddings:
    """Class to process PDF papers and create or update embeddings based on them.
    This class has an internal docs object that is used to store a list of PaperQA Doc objects.
    Each Doc object contains internally the embeddings of chunks of a single paper plus metadata with provenance (n_docs == n_processed_papers).

    Attributes:
        model (str): Generative Model that later will be used for answering queries.
        embedding_model (str): Encoder Model to use for creating the embeddings.
        output_embeddings_path (str): Path to save the output embeddings file.
        paper_directory (str): Directory containing the papers to process.
        docs (Docs, optional): Existing Docs object, to be able to manipulate it at Runtime.
    """
    def __init__(self,
                 model: str = "ollama/llama3.2:1b",
                 embedding_model: str = "ollama/mxbai-embed-large:latest",
                 output_embeddings_path: str = "out/embeddings.pkl",
                 paper_directory: str = "data/",
                 existing_docs: Docs = None,
                 ):
        self.output_embeddings_path = output_embeddings_path
        self.paper_directory = paper_directory
        self.embedding_model = embedding_model
        self.docs = existing_docs

        self.settings = configure_settings(model, self.embedding_model, self.paper_directory)

        emb_directory = os.path.dirname(self.output_embeddings_path)
        if emb_directory:
            os.makedirs(emb_directory, exist_ok=True)


    def __repr__(self):
        doc_len = len(self.docs.docs) if self.docs else 0
        chunk_len = len(self.docs.texts) if self.docs else 0
        return f"Embeddings(output_embeddings_path={self.output_embeddings_path}, total_docs={doc_len}, total_chunks={chunk_len})"

    def load_existing_embeddings(self, embeddings_path: str):
        """Load existing embeddings (List of Docs) from the specified path."""
        if os.path.exists(embeddings_path):
            self.docs = safely_load_pickle_file(embeddings_path)
            self.output_embeddings_path = embeddings_path
            print(f"Loaded existing embeddings from {embeddings_path}")
        else:
            print(f"No existing embeddings found at {self.output_embeddings_path}")
            self.docs = None


    def get_docs_details(self, verbose=False):
        """Get the details of the documents in the embeddings.
        This method retrieves the metadata of the documents stored in the embeddings.
        It returns a DataFrame with the details of each document.
        If verbose is True, it also prints the details of each document. 
        """
        def get_safe_key(doc_detail, key):
            try:
                return getattr(doc_detail, key)
            except AttributeError:
                return None

        if self.docs is None:
            print("No existing embeddings found.")
            return {}

        doc_details = []
        for doc_key in self.docs.docs.keys():
            doc_detail = self.docs.docs.get(doc_key)
            if doc_detail:
                details_dict = {
                    "doc_key": get_safe_key(doc_detail, "dockey"),
                    "key": get_safe_key(doc_detail, "key"),
                    "docname":get_safe_key(doc_detail, "docname"),
                    "title": get_safe_key(doc_detail, "title"),
                    "authors": get_safe_key(doc_detail, "authors"),
                    "year": get_safe_key(doc_detail, "year"),
                    "journal": get_safe_key(doc_detail, "journal"),
                    "volume": get_safe_key(doc_detail, "volume"),
                    "pages": get_safe_key(doc_detail, "pages"),
                    "doi": get_safe_key(doc_detail, "doi"),
                    "url": get_safe_key(doc_detail, "url"),
                    "file_location": get_safe_key(doc_detail, "file_location"),
                    "citation_count": get_safe_key(doc_detail, "citation_count"),
                    "source_quality": get_safe_key(doc_detail, "source_quality"),
                }
            doc_details.append(details_dict)
            # Print for informative purposes
            if verbose:
                for k, v in details_dict.items():
                    print(f"{k}: {v}")

        # Convert to DataFrame for better visualization
        details_df = pd.DataFrame(doc_details)
        return details_df

    def prepare_papers(self, paper_directory: str = None):
        """Takes a Path object with the location of PDFs to process and returns a list of the valid papers in the directory.
        Args:
            paper_directory (str, optional): location of (new) papers in PDF format. If None it takes the default paper directory

        Returns:
            ordered_file_list (str): list of the valid paper names in the given directory.
        """
        # Assign new paper directory if given. Otherswise, use the default one.
        if paper_directory is not None:
            self.paper_directory = paper_directory
        # Check that directory is found and list all files in it
        try:
            file_list = os.listdir(self.paper_directory)
        except FileNotFoundError:
            print(f"WARNING: Directory {self.paper_directory} not found.")
            return []

        # Drop None values from the ordered list if necessary
        ordered_file_list = [file for file in file_list if ".pdf" in file]
        return ordered_file_list

    def process_papers(self, paper_list: list[str]):
        """Transforms a list of valid PDFs into Doc Embeddings.
        The PDFs are chunked, processed and the embeddings are created using the Settings.
        The resulting embeddings are saved to the specified output path.

        Args:
            paper_list (list[str]): List of valid PDFs containing the papers that will be transformed into Embeddings.
        """
        
        # If Embeddings object has docs, we will add the new papers to it.
        if self.docs is not None:
            print("Adding new docs to the existing Embeddings.")
            docs = self.docs
        else:
            print("Creating new Docs object.")
            docs = Docs(index_path=self.output_embeddings_path)
        
        # Process each paper and add it to a PaperQA Docs object
        for i, doc in tqdm(enumerate(paper_list), total=len(paper_list), desc="Processing papers:", unit="paper"):
            doc_title = re.sub(" +", " ", doc.strip(".pdf").replace("_", " ").replace("-", " "))
            try:
                docs.add(
                        path=str(os.path.join(self.paper_directory, doc)),
                        docname=doc_title,
                        settings=self.settings
                )
                print(f"Correctly loaded {doc}")
            except Exception as e:
                print(f"Could not read {doc}: {e}")
                continue
        
        # Save the Docs object
        self.docs = docs
        with open(self.output_embeddings_path, "wb") as f:
            pkl.dump(docs, f)
        print(f"Docs object saved to {self.output_embeddings_path}")
    
    def get_document_chunks(self, docname: str, include_embeddings: bool = False, pages: list[int] = None):
        """Get the chunks of a document by its key.
        Args:
            docname (str): The key of the document to retrieve chunks from.
            include_embeddings (bool, optional): Whether to include the chunk vector in the returned chunks. Defaults to False.
            pages (list[int], optional): List of page numbers whose chunks will be returned (inclusive). If None, all chunks are returned.
        Returns:
            list: A list of chunks from the specified document.
        """
        if self.docs is None:
            print("No existing embeddings found.")
            return []

        if docname not in self.docs.docnames:
            print(f"Document with key {docname} not found in the embeddings.")
            return []

        chunks = []
        wanted_pages = set(pages) if pages is not None else None
        for doc_chunk in self.docs.texts:
            if doc_chunk.doc.docname == docname:
                page_nums = doc_chunk.name.split()[-1] if doc_chunk.name else "0"
                embedding = doc_chunk.embedding if include_embeddings else None
                if wanted_pages is not None:
                    # Check if the chunk's page number is in the specified range
                    chunk_page_ints = set([int(p) for p in page_nums.split("-")])
                    if len(wanted_pages.intersection(chunk_page_ints)) == 0:
                        continue
                # If pages is None, include all chunks
                chunks.append(DocumentChunk(
                    text=doc_chunk.text,
                    embedding=embedding,
                    docname=doc_chunk.doc.docname,
                    dockey=doc_chunk.doc.dockey,
                    chunkname=doc_chunk.name,
                    pages=page_nums
                ))

        return chunks