"""
tidyBib is a class designed to import and process bibliographic (.ris) files.

This class uses the ASReview library to load .ris files into a pandas DataFrame, 
providing methods to identify and resolve duplicates, detect entries with missing abstracts, 
and create a tidy bibliography. The tidy bibliography can be exported to a CSV file.

Attributes:
    ris (DataFrame): A pandas DataFrame containing the bibliographic data from the .ris file.

Methods:
    is_missing_abstract(abstract):
        check if an abstract is missing.
        
    resolve_duplicates(group):
        handle duplicates. It retains one entry per group, 
        preferring entries with an abstract.
        
    return_resolved_ris():
        Identifies and resolves duplicate entries in the bibliographic data.
        
    return_missing_abstracts():
        Identifies entries with missing abstracts.
        
    return_tidy_bib():
        Resolves duplicates, removes entries with missing abstracts, 
        and returns the cleaned/tidy bibliographic data.
        
    save_tidy_ris(output_file):
        Saves the tidy bib data to a CSV file.
"""

from asreview import ASReviewData
import pandas as pd

class tidyBib():
    def __init__(self, ris):
        # load .ris file using asreview
        self.ris = ASReviewData.from_file(ris).to_dataframe()
        
    @staticmethod
    def is_missing_abstract(abstract):
        if isinstance(abstract, str):
            return abstract.strip() == ''
        return pd.isna(abstract)
    
    @staticmethod
    def resolve_duplicates(group):
        # Handle duplicates: retain one entry, preferring the one with an abstract
        if group['abstract'].notna().any():
            return group.loc[group['abstract'].notna()].iloc[0]
        return group.iloc[0]
        

    def return_resolved_ris(self):
        # Identify and resolve duplicates
        resolved_ris = self.ris.groupby(['title', 'authors'], group_keys=False).apply(self.resolve_duplicates)
        return resolved_ris
    
    def return_missing_abstracts(self):
        # Identify entries with missing abstracts
        missing_abstracts = self.ris[self.ris['abstract'].apply(self.is_missing_abstract)]
        return missing_abstracts
        
        
    def return_tidy_bib(self):
        # identify and resolve duplicates
        # remove missing abstracts
        # return tidy bib
        
        resolved_ris = self.return_resolved_ris()
        tidy_ris = resolved_ris[~resolved_ris['abstract'].apply(self.is_missing_abstract)]
        
        return tidy_ris

    def save_tidy_ris(self, output_file):
        # save tidy bib
        tidy_ris = self.return_tidy_bib()
        tidy_ris.to_csv(output_file)
        return tidy_ris