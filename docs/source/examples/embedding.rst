Embedding Examples
==================

This page provides examples of creating embeddings for your document collection using MINE-DD.

Basic Embedding Example
-----------------------

The following example shows how to create embeddings for a collection of papers:

Note that to work with jupyter notebooks one needs to install `nest_asyncio` 
and import it into the notebook as follows:

.. code-block:: python
   %pip install nest_asyncio 
   
   import nest_asyncio
   nest_asyncio.apply()
   
1. Load Embeddings object

.. code-block:: python

   from minedd.embeddings import Embeddings
   from pathlib import Path
   
   # Set paths and parameters
   MODEL = "ollama/llama3.2"
   EMBEDDING = "ollama/mxbai-embed-large:latest"
   PAPERS_DIRECTORY = Path.home() / "papers_minedd/"
   
   # Create embeddings
   embeddings = Embeddings(
       model=MODEL,
       embedding_model=EMBEDDING,
       paper_directory=PAPERS_DIRECTORY,
       output_embeddings_path="my-embeddings.pkl"
   )
   
   embeddings

2. Load papers for processing

.. code-block:: python

   pdf_file_list = embeddings.prepare_papers()
   print(len(pdf_file_list))
   # Check the first 10 files
   pdf_file_list[:10]

3. Create/Load Embeddings and save in PKL

.. code-block:: python

   import os
   if os.path.exists("my-embeddings.pkl"):
       embeddings.load_existing_embeddings("my-embeddings.pkl")
   else:
       embeddings.process_papers(pdf_file_list)
       print("Embeddings created and saved to my-embeddings.pkl")

4. Inspect Documents Object

.. code-block:: python

   detail_df = embeddings.get_docs_details()
   detail_df

When to Create New Embeddings
-----------------------------

You should create new embeddings when:

* You have added new papers to your collection
* You want to use a different embedding model
* You want to update your existing embeddings

Using the CLI
-------------

You can also create embeddings using the command line interface:

.. code-block:: console

   minedd embed --paper_directory "/path/to/papers_minedd/" --embeddings_filename my-embeddings.pkl

