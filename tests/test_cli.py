"""Tests for the minedd.cli module."""
import sys
import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np

from minedd.cli import main, query_command, embed_command


class MockQuery:
    """Mock Query class for testing CLI functionality."""

    def __init__(self, model=None, embedding_model=None, paper_directory=None, output_dir=None):
        self.model = model
        self.embedding_model = embedding_model
        self.paper_directory = paper_directory
        self.output_dir = output_dir
        self.docs = None

    def load_embeddings(self, pickled_path):
        self.docs = MagicMock()
        return self

    def load_questions(self, file_path):
        return pd.DataFrame({
            'question': ['Test question 1', 'Test question 2'],
            'answer': [np.nan, np.nan],
            'context': [np.nan, np.nan],
            'citations': [np.nan, np.nan],
            'URL': [np.nan, np.nan]
        })

    def query_single(self, question, max_retries=2):
        return {
            'question': question,
            'answer': 'Test answer',
            'context': 'Test context',
            'citations': ['Citation 1', 'Citation 2'],
            'urls': ['URL 1', 'URL 2'],
            'raw_response': MagicMock()
        }

    def query_batch(self, questions, save_individual=True, output_file=None, max_retries=2):
        return pd.DataFrame({
            'question': ['Test question 1', 'Test question 2'],
            'answer': ['Test answer 1', 'Test answer 2'],
            'context': ['Test context 1', 'Test context 2'],
            'citations': [['Citation 1'], ['Citation 2']],
            'URL': [['URL 1'], ['URL 2']]
        })


@pytest.fixture
def mock_args_embed_papers():
    """Fixture for embed command."""
    args = MagicMock()
    args.llm = 'ollama/llama3.2:1b'
    args.embedding_model = 'ollama/mxbai-embed-large:latest'
    args.paper_directory = 'data/'
    args.output_dir = 'out'
    args.embeddings_filename = 'test_embeddings.pkl'
    return args

@pytest.fixture
def mock_args_single_question():
    """Fixture for args with single question."""
    args = MagicMock()
    args.llm = 'ollama/llama3.2:1b'
    args.embedding_model = 'ollama/mxbai-embed-large:latest'
    args.paper_directory = 'data/'
    args.output_dir = 'out'
    args.embeddings = 'test_embeddings.pkl'
    args.question = 'What is the meaning of life?'
    args.questions_file = None
    args.max_retries = 2
    return args


@pytest.fixture
def mock_args_batch_questions():
    """Fixture for args with batch questions."""
    args = MagicMock()
    args.llm = 'ollama/llama3.2:1b'
    args.embedding_model = 'ollama/mxbai-embed-large:latest'
    args.paper_directory = 'data/'
    args.output_dir = 'out'
    args.embeddings = 'test_embeddings.pkl'
    args.question = None
    args.questions_file = 'test_questions.xlsx'
    args.max_retries = 2
    return args


@pytest.fixture
def mock_args_no_question():
    """Fixture for args with no question."""
    args = MagicMock()
    args.llm = 'ollama/llama3.2:1b'
    args.embedding_model = 'ollama/mxbai-embed-large:latest'
    args.paper_directory = 'data/'
    args.output_dir = 'out'
    args.embeddings = 'test_embeddings.pkl'
    args.question = None
    args.questions_file = None
    args.max_retries = 2
    return args

@patch('minedd.cli.Query', MockQuery)
def test_query_command_single_question(mock_args_single_question, capsys):
    """Test query_command with a single question."""
    query_command(mock_args_single_question)

    # Capture stdout to check printed output
    captured = capsys.readouterr()

    # Check that the output contains expected sections
    assert "=== Question ===" in captured.out
    assert "What is the meaning of life?" in captured.out
    assert "=== Answer ===" in captured.out
    assert "Test answer" in captured.out
    assert "=== Sources ===" in captured.out
    assert "1. Citation 1" in captured.out
    assert "2. Citation 2" in captured.out
    assert "=== URLs ===" in captured.out
    assert "1. URL 1" in captured.out
    assert "2. URL 2" in captured.out


@patch('minedd.cli.Query', MockQuery)
def test_query_command_batch_questions(mock_args_batch_questions):
    """Test query_command with batch questions."""
    query_command(mock_args_batch_questions)
    # For batch questions, we primarily check that it completes without errors
    # We don't check output files since they're mocked


@patch('minedd.cli.Query', MockQuery)
def test_query_command_no_question(mock_args_no_question, capsys):
    """Test query_command with no question or questions file."""
    # This should exit with an error message
    with pytest.raises(SystemExit) as excinfo:
        query_command(mock_args_no_question)

    assert excinfo.value.code == 1

    captured = capsys.readouterr()
    assert "Error: Either --questions_file or --question must be provided." in captured.out


@patch('minedd.cli.argparse.ArgumentParser.parse_args')
@patch('minedd.cli.query_command')
def test_query_args_parsing(mock_query_command, mock_parse_args):
    """Test that argument parsing works correctly."""
    # Setup mock return for parse_args that simulates command line arguments
    args = MagicMock()
    args.command = "query"
    args.func = mock_query_command
    args.llm = 'ollama/custom:model'
    args.embedding_model = 'ollama/custom-embed:latest'
    args.paper_directory = 'custom_data/'
    args.output_dir = 'custom_out'
    args.embeddings = 'custom_embeddings.pkl'
    args.question = 'Custom question?'
    args.questions_file = None
    args.max_retries = 3
    mock_parse_args.return_value = args

    main()

    # Verify the arguments were passed to query_command
    mock_query_command.assert_called_once()
    called_args = mock_query_command.call_args[0][0]
    assert called_args.llm == 'ollama/custom:model'
    assert called_args.embedding_model == 'ollama/custom-embed:latest'
    assert called_args.paper_directory == 'custom_data/'
    assert called_args.output_dir == 'custom_out'
    assert called_args.embeddings == 'custom_embeddings.pkl'
    assert called_args.question == 'Custom question?'
    assert called_args.max_retries == 3


@patch('minedd.cli.argparse.ArgumentParser.parse_args')
@patch('minedd.cli.query_command')
def test_main_with_query_command(mock_query_command, mock_parse_args, mock_args_single_question):
    """Test the query command."""

    mock_args_single_question.command = "query"
    mock_args_single_question.func = mock_query_command
    mock_parse_args.return_value = mock_args_single_question

    # Call the main function
    main()

    # Verify that query_command was called with the mocked arguments
    mock_query_command.assert_called_once_with(mock_args_single_question)



class MockEmbeddings:
    def __init__(self, model=None, embedding_model=None, paper_directory=None, output_embeddings_path=None, existing_docs=None):
        self.model = model
        self.embedding_model = embedding_model
        self.paper_directory = paper_directory
        self.output_embeddings_path = output_embeddings_path
        self.docs = existing_docs

    def prepare_papers(self):
        return ['paper1.pdf', 'paper2.pdf']
    
    def process_papers(self, pdf_file_list):
        self.docs = MagicMock()
        return self
    
    def load_existing_embeddings(self, embeddings_path):
        self.docs = MagicMock()
        return self


@patch('minedd.cli.Embeddings', MockEmbeddings)
def test_embed_command(mock_args_embed_papers, capsys):
    """Test query_command with batch questions."""
    
    embed_command(mock_args_embed_papers)

    captured = capsys.readouterr()
    assert "Found 2 PDF files" in captured.out

    

@patch('minedd.cli.argparse.ArgumentParser.parse_args')
@patch('minedd.cli.embed_command')
def test_main_with_embed_command(mock_embed_command, mock_parse_args, mock_args_embed_papers):
    """Test the embed command."""
    # Mock the arguments for the embed command
    mock_args_embed_papers.command = "embed"
    mock_args_embed_papers.func = mock_embed_command

    mock_parse_args.return_value = mock_args_embed_papers

    # Call the main function
    main()

    # Verify that embed_command was called with the mocked arguments
    mock_embed_command.assert_called_once_with(mock_args_embed_papers)
