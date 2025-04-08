import os
import pandas as pd
from difflib import get_close_matches
from tqdm import tqdm
from paperqa import Docs
import pickle as pkl
from pathlib import Path
"""Class to handle embeddings and process papers.

Attributes:
    output_embeddings_path (str): Path to save the output embeddings file.
    docs (Docs): Existing Docs object, to be able to manipulate it at Runtime.

Methods:
    __init__(output_embeddings_path, existing_docs): Initializes the Embeddings class.
    prepare_papers(papers_directory): Prepares the list of papers in the given directory.
    process_papers(settings, papers_directory, paper_list): Processes and adds papers to the Docs object.
"""
class Embeddings:
    def __init__(self,
                 output_embeddings_path: str = "embeddings.pkl",
                 existing_docs: Docs = None,
                 ):
        self.output_embeddings_path = output_embeddings_path
        self.docs = existing_docs


    def __repr__(self):
        #TODO: If possible show the Dimensions of the embeddings table [n_docs, dimensions]
        return f"Embeddings(output_embeddings_path={self.output_embeddings_path}, docs={self.docs})"

    def load_existing_embeddings(self, embeddings_path: str):
        """Load existing embeddings from the specified path."""
        if os.path.exists(embeddings_path):
            with open(embeddings_path, "rb") as f:
                self.docs = pkl.load(f)
            self.output_embeddings_path = embeddings_path
            print(f"Loaded existing embeddings from {embeddings_path}")
        else:
            print(f"No existing embeddings found at {self.output_embeddings_path}")
            self.docs = None

    def prepare_papers(self, papers_directory: Path):
        """_summary_

        Args:
            papers_directory (Path): _description_

        Returns:
            _type_: _description_
        """
        file_list = os.listdir(papers_directory)
        path_files = [os.path.join(papers_directory, f) for f in file_list]

        path_df = pd.DataFrame(
            {
                "PDF Name": file_list,
                "Path": path_files
            }
        )
        # Order file list by matching names
        ordered_file_list = []
        for title in path_df['PDF Name']:
            closest_match = get_close_matches(title, file_list, n=1)
            ordered_file_list.append(closest_match[0] if closest_match else None)

        # Drop None values from the ordered list if necessary
        ordered_file_list = [file for file in ordered_file_list if file is not None]

        return ordered_file_list

    def process_papers(self, settings, papers_directory: Path, paper_list: list[str] = None):
        """Process and add papers to the Docs object."""

        if paper_list is not None:
            ordered_file_list = paper_list
        else:
            ordered_file_list = self.prepare_papers(papers_directory)

        if self.docs is not None:
            print("Adding new docs to the existing Embeddings.")
            docs = self.docs
        else:
            print("Creating new Docs object.")
            docs = Docs(index_path=self.output_embeddings_path)

        for i, doc in tqdm(enumerate(ordered_file_list), total=len(ordered_file_list)):
            try:
                docs.add(
                        path=str(os.path.join(papers_directory, doc)),
                        docname=doc,
                        settings=settings
                )
                print(f"Correctly loaded {doc}")
            except Exception as e:
                print(f"Could not read {doc}: {e}")
                continue

        self.docs = docs
        # Save the Docs object
        with open(self.output_embeddings_path, "wb") as f:
            pkl.dump(docs, f)
        print(f"Docs object saved to {self.output_embeddings_path}")