"""
Query module for the MINEDD package.

This module provides functionality to query document collections
created with the embeddings module.
"""

import pickle as pkl
import os
import pandas as pd
import numpy as np
import pathlib
import platform
import re
import subprocess
import time

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
        paper_directory: str = "data/",
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
        self.settings = configure_settings(model, embedding_model, paper_directory)

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

        if not os.path.exists(pickled_path):
            raise FileNotFoundError(f'File {pickled_path} not found')

        # Windows users do not support PosixPath (whose references are contained in pickle files)
        # so when unpickling the embeddings, Python tries to instantiate these PosixPath objects, but fails
        # since they're not supported

        # one way to fix this is to point the PosixPath point to WindowsPath temporarily during unpickling.
        # from: "https://stackoverflow.com/questions/57286486/i-cant-load-my-model-because-i-cant-put-a-posixpath"

        system = platform.system()

        if system == "Windows":
            # Save original PosixPath
            temp = pathlib.PosixPath

            try:
                pathlib.PosixPath = pathlib.WindowsPath  # point to the windows path
                with open(pickled_path, 'rb') as f:
                    self.docs = pkl.load(f)
            finally:
                # restore the original posixpath
                pathlib.PosixPath = temp
        else:
            with open(pickled_path, 'rb') as f:
                self.docs = pkl.load(f)

        return self

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

    # sometimes models are not available (i.e. must be downloaded first)
    def _ensure_model_available(self, model_name=None):
        """
        Ensure that the model is available in Ollama.
        If not, tries to pull it.

        Args:
            model_name (str, optional): Name of the model. If None, uses self.model

        Returns:
            bool: True if model is available, False otherwise

        Raises:
            RuntimeError: If pulling fails
        """
        if model_name is None:
            # get model name from self.model
            model_name = self.model.split('/')[-1] # format here always starts with "ollama/"

        try:
            # check that model is listed
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                check=True
            )

            if model_name in result.stdout:
                return True

            print(f"Model {model_name} not found. Attempting to pull it ...")
            # Run the pull command without assigning the unused result
            subprocess.run(
                ["ollama", "pull", model_name],
                capture_output=True,
                text=True,
                check=True
            )

            print(f"Successfully pulled {model_name}")
            return True

        except subprocess.CalledProcessError as e:
            print(f"Error with Ollama: {e}")
            print(f"Subprocess output: {e.stdout}")
            print(f"Subprocess error: {e.stderr}")
            return False

    # I have divided the query into single and batch.
    # query_single is designed for interactive usage where you want immediate answers (useful if making the interactions real time)
    # query_batch is instead for multiple questions, when these must be submitted as a job.sh on Snellius with file saving
    def query_single(self, question: str, max_retries: int = 2) -> dict:
        """
        Query the documents with a single question.

        Args:
            question: The question to ask
            max_retries: Maximum number of retried if model loading fails

        Returns:
            dict: A dictionary containing the answer, context, and citations

        Raises:
            ValueError: If embeddings haven't been loaded
        """
        if self.docs is None:
            raise ValueError('No document embeddings loaded. Call load_embeddings()')

        retries = 0
        while retries <= max_retries:
            try:
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

            except Exception as e:
                error_str = str(e)
                # check if error is about model not found
                if ("model not found" in error_str.lower() or
                "OllamaException" in error_str or
                "model '" in error_str and "' not found" in error_str):
                    match = re.search(r"model '([^']+)' not found", error_str )
                    if match:
                        model_name = match.group(1)
                    else:
                        model_name = self.model.split('/')[-1]

                    print("Model not found error detected. Check model availability..")

                    if self._ensure_model_available(model_name):
                        print(f"Model {model_name} is now available. Retrying query ({retries+1}/{max_retries})")
                        retries += 1
                        time.sleep(2) # allows time for ollama background processes
                        continue
                    else:
                        raise RuntimeError(f"Failed to pull {model_name}. Model not available.")

            if retries >= max_retries:
                print(f"Error after {retries} retries: {error_str}")
                raise
            print(f"Error: {error_str}")
            print(f"Retrying ({retries + 1}/{max_retries})...")
            retries += 1
            time.sleep(2)


    def query_batch(
        self,
        questions: list[str] | pd.DataFrame,
        save_individual: bool = True,
        output_file: str | None = None,
        max_retries: int = 2
    ) -> pd.DataFrame:
        """
        Query the documents with multiple questions.

        Args:
            questions: Either a list of question strings or a DataFrame with a 'question' column
            save_individual: Whether to save individual query results
            output_file: Path to save the complete results DataFrame

        Returns:
            DataFrame: A pandas DataFrame with questions and their answers

        Raises:
            ValueError: If embeddings haven't been loaded
        """
        if self.docs is None:
            raise ValueError("No document embeddings loaded. Call load_embeddings() first.")

        # Convert list to DataFrame if needed
        if isinstance(questions, list):
            questions_df = pd.DataFrame({'question': questions})
        else:
            questions_df = questions.copy()

        # Get model name for filenames: Replace any characters that are invalid in filenames
        model_name = self.model.split('/')[-1].replace(':', '_').replace('/', '_')

        # Flag to track if we've already ensured model availability
        model_checked = False

        # Process each question
        for q_idx in range(len(questions_df)):
            print(f'Processing question {q_idx + 1}/{len(questions_df)}')

            question = questions_df.loc[q_idx, 'question']

            retries = 0
            while retries <= max_retries:

                try:
                    answer_obj = self.docs.query(question, settings=self.settings)

                    # Store results in DataFrame
                    questions_df.at[
                        q_idx, 'answer'] = answer_obj.formatted_answer if answer_obj.formatted_answer else np.nan
                    questions_df.at[q_idx, 'context'] = answer_obj.context if answer_obj.context else np.nan
                    questions_df.at[q_idx, 'citations'] = [
                        context.text.doc.citation for context in answer_obj.contexts
                        if hasattr(context.text.doc, 'citation')
                    ]
                    questions_df.at[q_idx, 'URL'] = [
                        context.text.doc.url for context in answer_obj.contexts
                        if hasattr(context.text.doc, 'url')
                    ]

                    # Save individual answer (if requested)
                    if save_individual:
                        output_path = f"{self.output_dir}/answer_{q_idx}_{model_name}.pkl"
                        with open(output_path, "wb") as f:
                            pkl.dump(answer_obj, f)
                        print(f"Answer {q_idx} saved to {output_path}")
                    break

                except Exception as e:
                    print(f"Error processing question {q_idx}: {e}")
                    error_str = str(e)
                    if "model not found" in error_str.lower() and not model_checked:
                        used_model = self.model.split('/')[-1]
                        print("Model not found error detected. Check model availability..")

                        if self._ensure_model_available(used_model):
                            model_checked = True
                            print(f"Model {used_model} available. Retrying query ({retries+1}/{max_retries})")
                            retries += 1
                            time.sleep(2)
                            continue
                        else:
                            questions_df.at[q_idx, 'answer'] = f"Error: model {used_model} not available"
                            break
                    if retries >= max_retries:
                        print(f"Error after {retries} retries: {error_str}")
                        questions_df.at[q_idx, 'answer'] = f"Error: {error_str}"
                        break
                    print(f"Error processing question {q_idx}: {e}")
                    print(f"Retrying ({retries+1}/{max_retries})...")
                    retries += 1
                    time.sleep(2)

        # Save complete results if requested
        if output_file:
            questions_df.to_excel(output_file, index=False)
            print(f"Results saved to {output_file}")

        return questions_df



