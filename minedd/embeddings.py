import os
import re
import pandas as pd
from difflib import get_close_matches
from tqdm import tqdm
from paperqa import Docs
import pickle as pkl

class Embedding:
    def __init__(self,
                 papers_metadata_file: str,
                 papers_directory: str,
                 embeddings_directory: str,
                 output_embeddings_path: str = "embeddings.pkl"
                 ):
        self.papers_metadata_file = papers_metadata_file
        self.papers_directory = papers_directory
        self.embeddings_directory = embeddings_directory
        self.output_embeddings_path = output_embeddings_path
    
    def prepare_metadata(self):
        """Prepare metadata for paper processing."""
        papers = pd.read_csv(self.papers_metadata_file)
        file_list = os.listdir(self.papers_directory)
        path_files = [os.path.join(self.papers_directory, f) for f in file_list]

        path_df = pd.DataFrame(
            {
                "PDF Name": file_list,
                "Path": path_files
            }
        )

        # Merge metadata with file paths
        papers_metadata = papers.merge(path_df, how='left', on='PDF Name')
        papers_metadata = papers_metadata.dropna(subset=['Path'])

        # Order file list by matching names
        ordered_file_list = []
        for title in papers_metadata['PDF Name']:
            closest_match = get_close_matches(title, file_list, n=1)
            ordered_file_list.append(closest_match[0] if closest_match else None)

        # Drop None values from the ordered list if necessary
        ordered_file_list = [file for file in ordered_file_list if file is not None]

        return papers_metadata, ordered_file_list

    def print_doi(self, doi):
        """Extract DOI from a string."""
        doi_pattern = r"10.\d{4,9}/[-._;()/:A-Za-z0-9]+"
        doi_match = re.search(doi_pattern, doi)
        if doi_match:
            return doi_match.group(0)
        print("No valid DOI has been found")
        return None

    def process_papers(self, settings, ordered_file_list, papers_metadata=None):
        """Process and add papers to the Docs object."""
        docs = Docs(index_path=self.output_embeddings_path)

        for i, doc in tqdm(enumerate(ordered_file_list), total=len(ordered_file_list)):
            try:
                if papers_metadata:
                    docs.add(
                        path=str(os.path.join(self.papers_directory, doc)),
                        doi=self.print_doi(papers_metadata['DOI'].iloc[i]),
                        docname=doc,
                        authors=papers_metadata['Authors'].iloc[i],
                        title=papers_metadata['Title'].iloc[i],
                        settings=settings
                    )
                else: 
                    docs.add(
                        path=str(os.path.join(self.papers_directory, doc)),
                        docname=doc,
                        settings=settings
                    )
                print(f"Correctly loaded {doc}")
            except Exception as e:
                print(f"Could not read {doc}: {e}")
                continue

        # Save the Docs object
        with open(self.output_embeddings_path, "wb") as f:
            pkl.dump(docs, f)
        print(f"Docs object saved to {self.output_embeddings_path}")