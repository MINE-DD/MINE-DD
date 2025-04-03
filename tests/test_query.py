"""Tests for the minedd.query module."""
import os
import pytest
import pandas as pd
from minedd.query import Query


def test_load_embeddings_success():
    """Test successful loading of embeddings from a pickle file."""
    # Create query instance


    query = Query()

    # Load existing test embeddings
    test_path = os.path.join(os.path.dirname(__file__), "data/test_embeddings.pkl")

    # Verify files exists
    assert os.path.exists(test_path), f"Test embeddings pickle file not found at {test_path}"

    # Load embeddings
    result = query.load_embeddings(test_path)

    # Verify docs are loaded and method returns self
    assert query.docs is not None, "Embeddings should be loaded"
    assert result is query, "Method should return self for chaining"


def test_load_embeddings_file_not_found():
    """Test load_embeddings raises FileNotFoundError when file doesn't exist."""
    # Create query instance
    query = Query()
    nonexistent_path = "nonexistent_file.pkl"

    # Verify correct error is raised
    with pytest.raises(FileNotFoundError) as excinfo:
        query.load_embeddings(nonexistent_path)
    assert f"File {nonexistent_path} not found" in str(excinfo.value)

# This verifies the function works correctly
# with complete input data.
def test_load_questions_existing_columns():
    """Test loading questions from Excel file with all required columns."""
    # Create query instance
    query = Query()

    # Path to test file with all columns
    test_path = os.path.join(os.path.dirname(__file__), "data/excel/test_questions_with_columns.xlsx")

    # Verify file exists
    assert os.path.exists(test_path), f"Test Excel file not found at {test_path}"

    # Load questions
    result = query.load_questions(test_path)

    # Verify the result is a DataFrame with the correct questions
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2
    assert 'question' in result.columns
    assert 'answer' in result.columns
    assert 'context' in result.columns
    assert 'citations' in result.columns
    assert 'URL' in result.columns
    assert result['question'].tolist() == ['What is MINEDD?', 'How does embedding work?']

    # Verify all columns have object dtype for handling lists
    assert result['answer'].dtype == 'object'
    assert result['context'].dtype == 'object'
    assert result['citations'].dtype == 'object'
    assert result['URL'].dtype == 'object'


# This instead tests when the excel file has the question column but is missing other columns
# the function should properly add missing columns so the method below tests that this is done
def test_load_questions_missing_columns():
    """Test loading questions from Excel file with missing columns."""
    # Create query instance
    query = Query()

    # Path to test file with only question column
    test_path = os.path.join(os.path.dirname(__file__), "data/excel/test_questions.xlsx")

    # Verify file exists
    assert os.path.exists(test_path), f"Test Excel file not found at {test_path}"

    # Load questions
    result = query.load_questions(test_path)

    # Verify the result is a DataFrame with all columns added
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2
    assert 'question' in result.columns
    assert 'answer' in result.columns
    assert 'context' in result.columns
    assert 'citations' in result.columns
    assert 'URL' in result.columns
    assert result['question'].tolist() == ['What is MINEDD?', 'How does embedding work?']

    # Verify columns were added with nan values and proper dtype
    assert pd.isna(result['answer']).all()
    assert pd.isna(result['context']).all()
    assert pd.isna(result['citations']).all()
    assert pd.isna(result['URL']).all()
    assert result['answer'].dtype == 'object'
    assert result['context'].dtype == 'object'
    assert result['citations'].dtype == 'object'
    assert result['URL'].dtype == 'object'
