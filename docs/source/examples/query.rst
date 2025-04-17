Query Examples
==============

This page provides examples of querying your document collection using MINE-DD.

Setting Up
----------

Before querying, make sure you have embeddings for your documents. See the :doc:`embedding` page for details on creating embeddings.

For Jupyter notebooks, you'll need to install and apply ``nest_asyncio``:

.. code-block:: python

   %pip install pydantic==2.10.1
   %pip install nest_asyncio
   
   import nest_asyncio
   nest_asyncio.apply()
   import pandas as pd
   
Basic Setup
-----------

Start by importing the ``Query`` class and setting up your environment:

.. code-block:: python

   from minedd.query import Query
   import os
   
   # Define your models and directories
   MODEL = "ollama/llama3.2:1b"
   EMBEDDING = "ollama/mxbai-embed-large:latest"
   PAPERS_DIRECTORY = "path/to/your/papers/"
   OUTPUT_DIR = "out"
   
   # Initialize the Query object
   query = Query(
       model=MODEL,
       embedding_model=EMBEDDING,
       paper_directory=PAPERS_DIRECTORY,
       output_dir=OUTPUT_DIR
   )
   
   # Load pre-generated embeddings
   embeddings_path = "path/to/your/embeddings.pkl"
   query.load_embeddings(embeddings_path)

Single Question Query
---------------------

To query with a single question:

.. code-block:: python

   # Define your question
   question = "What is Ribulose bisphosphate carboxylase?"
   
   result = query.query_single(question)
   
   # Display the answer
   print("\nAnswer:", result['answer'])

The response will include an answer sourced from your document collection, along with citations and context information.

Batch Processing
----------------

For multiple questions, you can process them as a batch using a dataframe:

.. code-block:: python

   # Load questions from Excel file
   questions = pd.read_excel('path/to/your/questions.xlsx')
   
   # Process all questions and save results
   results_df = query.query_batch(
       questions=questions,
       save_individual=True,
       output_file=f"{OUTPUT_DIR}/batch_results.xlsx"
   )
   
   # View results
   results_df.head()

The results dataframe contains columns for questions, answers, context information, and citations.

Using the CLI
-------------

You can also perform queries using the command-line interface:

.. code-block:: console

   minedd query --model "ollama/llama3.2:1b" --embedding_model "ollama/mxbai-embed-large:latest" --embeddings_path "path/to/embeddings.pkl" --paper_directory "path/to/papers/" --question "What is X?" --output_dir "out/"
