"""Integration tests for minedd.query that require Ollama and local GPU.

These tests verify the full functionality of query_single and query_batch methods,
but require a local environment with Ollama running. These tests are skipped
in CI environments.

To run these tests locally:
1. Make sure Ollama is running (`ollama serve`)
2. Run: `pytest tests/test_query_integration.py -v` or `pytest -m "integration" -v`
"""

import os
import pytest
import pandas as pd
import numpy as np
from minedd.query import Query

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration

# Skip these tests in CI environments
if os.environ.get("CI") == "true" or os.environ.get("GITHUB_ACTIONS") == "true":
    pytestmark = pytest.mark.skip(reason="Integration tests requiring Ollama - skipped in CI environment")

# Global flag to optionally skip even in local environments
# Set to False when you want to run the integration tests locally
SKIP_OLLAMA_TESTS = False

if SKIP_OLLAMA_TESTS:
    pytestmark = pytest.mark.skip(reason="Ollama integration tests explicitly disabled with SKIP_OLLAMA_TESTS=True")


class TestQueryIntegration:
    """Integration tests for the Query class that require Ollama."""

    @pytest.fixture(scope="class")
    def query_engine(self):
        """Create a Query instance for testing."""
        engine = Query(
            model="ollama/llama3.2:1b",  # Use a small model for faster tests
            embedding_model="ollama/mxbai-embed-large:latest",
            paper_directory="tests/data/",
            output_dir="tests/out"
        )

        # Load test embeddings
        test_embeddings_path = os.path.join(os.path.dirname(__file__), "data/test_embeddings.pkl")
        assert os.path.exists(test_embeddings_path), f"Test embeddings not found at {test_embeddings_path}"
        engine.load_embeddings(test_embeddings_path)

        return engine

    @pytest.fixture
    def sample_question(self):
        """Provide a sample question for testing."""
        return "What is the main topic of this paper?"

    def test_query_single(self, query_engine, sample_question):
        """Test querying with a single question."""
        # Run the query
        print("\n\n===== RUNNING QUERY SINGLE TEST =====")
        print(f"Question: {sample_question}")

        try:
            result = query_engine.query_single(sample_question)
            print("Query completed successfully")
        except Exception as e:
            print(f"ERROR executing query: {e}")
            import traceback
            traceback.print_exc()
            raise

        # Verify result structure
        assert isinstance(result, dict), "Result should be a dictionary"
        assert 'question' in result, "Result should contain 'question' key"
        assert 'answer' in result, "Result should contain 'answer' key"
        assert 'context' in result, "Result should contain 'context' key"
        assert 'citations' in result, "Result should contain 'citations' key"
        assert isinstance(result['citations'], list), "Citations should be a list"

        # Verify question matches input
        assert result['question'] == sample_question

        # Verify answer is not empty
        assert result['answer'] is not None and result['answer'] != ""

        # Verify answer doesn't contain the word 'error' (case insensitive)
        answer_text = str(result['answer']).lower()
        assert 'error' not in answer_text, f"Answer contains the word 'error': {result['answer']}"

    def test_query_batch(self, query_engine):
        """Test querying with a batch of questions."""
        print("\n\n===== RUNNING QUERY BATCH TEST =====")

        # Create a test output file path
        output_file = os.path.join(os.path.dirname(__file__), "out/test_batch_results.xlsx")
        print(f"Output file will be saved to: {output_file}")
        
        # Load questions from the Excel file
        test_file_path = os.path.join(os.path.dirname(__file__), "data/excel/test_questions.xlsx")
        assert os.path.exists(test_file_path), f"Test Excel file not found at {test_file_path}"
        print(f"Loading test questions from: {test_file_path}")
        
        # Load and prepare questions
        questions_df = pd.read_excel(test_file_path)
        
        # Ensure required columns exist with appropriate dtypes
        for col in ['answer', 'context', 'citations', 'URL']:
            if col not in questions_df.columns:
                questions_df[col] = np.nan
                questions_df[col] = questions_df[col].astype(object)

        # Print the questions
        print("\nQuestions:")
        for i, question in enumerate(questions_df['question']):
            print(f"{i+1}. {question}")

        # Run the query batch
        print("\nProcessing batch queries...")
        try:
            result_df = query_engine.query_batch(
                questions_df,
                save_individual=True,
                output_file=output_file
            )
            print("Batch query completed successfully")
        except Exception as e:
            print(f"ERROR executing batch query: {e}")
            import traceback
            traceback.print_exc()
            raise

        # Print results for each question
        print("\n===== BATCH RESULTS =====")
        for i in range(len(result_df)):
            print(f"\nQuestion {i+1}: {result_df.loc[i, 'question']}")
            print(f"Answer: {result_df.loc[i, 'answer']}")

            # Print citations if available
            citations = result_df.loc[i, 'citations']
            if isinstance(citations, list) and citations:
                print("\nCitations:")
                for j, citation in enumerate(citations):
                    print(f"  {j+1}. {citation}")
            elif not pd.isna(citations).all() if hasattr(citations, 'all') else not pd.isna(citations):
                print(f"\nCitation: {citations}")

            # Print URLs if available
            urls = result_df.loc[i, 'URL']
            if isinstance(urls, list) and urls:
                print("\nURLs:")
                for j, url in enumerate(urls):
                    print(f"  {j+1}. {url}")
            elif not pd.isna(urls).all() if hasattr(urls, 'all') else not pd.isna(urls):
                print(f"\nURL: {urls}")

            print("\n---")

        print("\n============================\n")

        # Verify result is a DataFrame with the same number of rows
        assert isinstance(result_df, pd.DataFrame), "Result should be a DataFrame"
        assert len(result_df) == len(questions_df), "Result should have same number of rows as input"

        # Verify all required columns are present
        assert 'question' in result_df.columns, "Result should contain 'question' column"
        assert 'answer' in result_df.columns, "Result should contain 'answer' column"
        assert 'context' in result_df.columns, "Result should contain 'context' column"
        assert 'citations' in result_df.columns, "Result should contain 'citations' column"

        # Verify answers were generated
        assert not pd.isna(result_df['answer']).all(), "Some answers should be generated"

        # Verify output file was created
        assert os.path.exists(output_file), f"Output file not created at {output_file}"
        print(f"\nOutput file created successfully at {output_file}")

        # Clean up
        if os.path.exists(output_file):
            os.remove(output_file)
            print("Output file cleaned up")

        # Clean up individual answer files
        cleaned_files = 0
        # Check both possible filename formats
        for idx in range(len(questions_df)):
            # Format from model="ollama/llama3.2:1b" becomes llama3.2_1b
            answer_file1 = f"tests/out/answer_{idx}_llama3.2_1b.pkl"

            if os.path.exists(answer_file1):
                os.remove(answer_file1)
                cleaned_files += 1

        print(f"Cleaned up {cleaned_files} individual answer files")
