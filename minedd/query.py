"""
Query module for the MINEDD package.

This module provides functionality to query document collections
created with the embeddings module.
"""

import pickle as pkl
import os
import pandas as pd
import numpy as np
from paperqa.settings import Settings, AgentSettings
from paperqa import Docs

from minedd.utils import configure_settings

class Query:
    """
    A class to query document collections using PaperQA.

    This class provides an interface to load document embeddings,
    query them with natural language questions, and save the results.
    """

    def __init__(
        self,
        model: str = "ollama/llama3.2:1b",
        embedding_model: str = "ollama/mxbai-embed-large:latest",
        output_dir: str = "out"
    ):
        """
        Initialize the Query.

        Args:
            model: The LLM model to use for answering queries
            embedding_model: The embedding model to use
            output_dir: Directory to save outputs
        """
        self.model = model
        self.embedding_model = embedding_model
        self.output_dir = output_dir
        self.docs = None
        self.settings = configure_settings(model, embedding_model)

        # create output directory if this doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

    def load_embeddings(self, pickled_path: str) -> 'Query':
        """
        Load document embeddings from a pickle file.

        Args:
            pickle_path: Path to the pickled Docs object

        Returns:
            self: For method chaining

        Raises:
            FileNotFoundError: If the pickle file is not found
        """
        try:
            with open(pickled_path, 'rb') as f:
                self.docs = pkl.load(f)
            return self
        except FileNotFoundError:
            raise FileNotFoundError(f'File {pickled_path} not found')

    def load_questions(self, file_path: str) -> pd.DataFrame:
        """
        Load questions from an Excel file.

        Args:
            file_path: Path to the Excel file containing questions

        Returns:
            DataFrame: A pandas DataFrame with questions and columns for answers
        """
        questions = pd.read_excel(file_path)
        for col in ['answer', 'context', 'citations', 'URL']:
            if col not in questions.columns:
                questions[col] = np.nan
                # keys must contain a list to we need to change the type to object
                questions[col] = questions[col].astype(object)
            return questions

    # I have divided the query into single and batch.
    # query_single is designed for interactive usage where you want immediate answers -- perhaps making the interactions real time
    # query_batch is instead for multiple questions, when these must be submitted as a job.sh with file saving
    def query_single(self, question: str) -> dict:
        """
        Query the documents with a single question.

        Args:
            question: The question to ask

        Returns:
            dict: A dictionary containing the answer, context, and citations

        Raises:
            ValueError: If embeddings haven't been loaded
        """
        if self.docs is None:
            raise ValueError('No document embeddings loaded. Call load_embeddings()')

        answer_obj = self.docs.query(question, settings=self.settings)

        result = {
            'question': question,
            # sometimes lists are empty -- added np.NaN to catch that
            'answer': answer_obj.formatted_answer if answer_obj.formatted_answer else np.nan,
            'context': answer_obj.context if answer_obj.context else np.nan,
            'citations': [
                context.text.doc.citation for context in answer_obj.contexts
                if hasattr(context.text.doc, 'citation')
            ],
            'urls': [
                context.text.doc.url for context in answer_obj.contexts
                if hasattr(context.text.doc, 'url')
            ],
            'raw_response': answer_obj
        }
        return result

