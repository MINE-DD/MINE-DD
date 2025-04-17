Welcome to MINE-DD's documentation!
===================================

**MINE-DD** (Mining the past to protect against Diarrheal Disease in the future) is a collaborative 
research project between the eScience Center and Amsterdam UMC. The project focuses on addressing 
the global health challenge of diarrheal disease in the context of climate change.

MINE-DD is a Python package that leverages artificial intelligence to extract and synthesize 
insights about climate's impact on diarrheal diseases from scientific literature. It enables 
researchers to efficiently query and analyze large collections of academic papers that would be 
impractical to read manually.

The package:

* Takes a collection of scientific papers (PDFs)
* Processes them to create embeddings (vector representations)
* Allows users to query these papers with natural language questions
* Returns answers with citations and context from the relevant papers

.. note::

   This project is under active development.

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   usage
   api
   
.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples/embedding
   examples/query
   
.. toctree::
   :maxdepth: 1
   :caption: Development

   https://github.com/MINE-DD/MINE-DD/blob/main/CONTRIBUTING.md
   https://github.com/MINE-DD/MINE-DD/blob/main/CHANGELOG.md
