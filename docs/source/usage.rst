Usage
=====

.. _installation:

Installation
------------

Requirements
^^^^^^^^^^^

* Python 3.11 or higher
* `Ollama <https://ollama.ai/>`_ installed locally for running LLMs and embeddings

Installation Steps
^^^^^^^^^^^^^^^^^

1. Clone the repository:

.. code-block:: console

   git clone git@github.com:MINE-DD/MINE-DD.git
   cd MINE-DD

2. Install the package:

.. code-block:: console

   # Standard installation
   python -m pip install .

   # Development installation
   python -m pip install -e ".[dev]"

Before Using MINE-DD
------------------

Make sure Ollama is running in the background:

.. code-block:: console

   ollama serve

Creating Paper Embeddings
-----------------------

Use the ``minedd embed`` command to create embeddings based on your document collection:

.. code-block:: console

   minedd embed --paper_directory "/path/to/papers_minedd/" --embeddings_filename my-embeddings.pkl

Available Parameters:

* ``--embeddings_filename``: Name the embeddings pickle file where the index is saved
* ``--output_dir``: Directory to save the embeddings pkl (default: 'out')
* ``--embedding_model``: Embedding model (default: 'ollama/mxbai-embed-large:latest')
* ``--paper_directory``: Directory with paper files (default: 'data/')
* ``--augment_existing``: If True it will add new documents to the existing pkl file provided. Otherwise it creates the pkl file from scratch.

Querying Papers
-------------

Use the ``minedd query`` command to ask questions about your document collection:

.. code-block:: console

   minedd query --embeddings embeddings/papers_embeddings.pkl --questions_file questions.xlsx --output_dir results/

or for a single question:

.. code-block:: console

   minedd query --embeddings embeddings/papers_embeddings.pkl --question "What is the relationship between climate change and diarrheal disease?"

Available Parameters:

* ``--embeddings``: Path to the embeddings pickle file (required)
* ``--questions_file``: Path to Excel file with questions
* ``--question``: Single question to ask
* ``--llm``: LLM model to use (default: 'ollama/llama3.2:1b')
* ``--embedding_model``: Embedding model (default: 'ollama/mxbai-embed-large:latest')
* ``--paper_directory``: Directory with paper files (default: 'data/')
* ``--output_dir``: Directory to save outputs (default: 'out')
* ``--max_retries``: Retries for model loading failures (default: 2)

Testing
-------

The project includes two types of tests:

1. **Standard Tests**: These run in CI environments (GitHub Actions) and don't require Ollama or GPU access.

   .. code-block:: console

      # Run all standard tests
      pytest
      
      # Run specific test files
      pytest tests/test_utils.py tests/test_cli.py

2. **Integration Tests**: These test the full functionality including LLM queries with Ollama, requiring a local environment with Ollama running.

   .. code-block:: console

      # Enable integration tests by setting SKIP_OLLAMA_TESTS=False in tests/test_query_integration.py
      # Then run:
      pytest -m integration

