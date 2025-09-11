[![github repo badge](https://img.shields.io/badge/github-repo-000.svg?logo=github&labelColor=gray&color=blue)](https://github.com/MINE-DD/mine-dd)
[![github license badge](https://img.shields.io/github/license/MINE-DD/mine-dd)](https://github.com/MINE-DD/mine-dd )
[![RSD](https://img.shields.io/badge/rsd-mine_dd-00a3e3.svg)](https://research-software-directory.org/projects/mine-dd)
[![fair-software badge](https://img.shields.io/badge/fair--software.eu-%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8B-yellow)](https://fair-software.eu)
[![build](https://github.com/MINE-DD/mine-dd/actions/workflows/build.yml/badge.svg)](https://github.com/MINE-DD/mine-dd/actions/workflows/build.yml)
[![cffconvert](https://github.com/MINE-DD/mine-dd/actions/workflows/cffconvert.yml/badge.svg)](https://github.com/MINE-DD/mine-dd/actions/workflows/cffconvert.yml)
[![markdown-link-check](https://github.com/MINE-DD/mine-dd/actions/workflows/markdown-link-check.yml/badge.svg)](https://github.com/MINE-DD/mine-dd/actions/workflows/markdown-link-check.yml)

## MINE-DD

(Mining the past to protect against Diarrheal Disease in the future) is a collaborative research project between the eScience Center, the University of Amsterdam (UvA) and Amsterdam UMC. 
The project focuses on addressing the global health challenge of diarrheal disease in the context of climate change.

## Description

MINE-DD is a Python package that leverages artificial intelligence to extract and synthesize insights about climate's impact on diarrheal diseases from scientific literature. 
It enables researchers to efficiently query and analyze large collections of academic papers that would be impractical to read manually. Built on the [PaperQA2](https://github.com/neuracap/paperqa) framework, 
the package implements an advanced question-answering system that provides detailed, citation-backed responses, ensuring that every insight is directly traceable to its original source materials.
The package:

- Takes a collection of scientific papers (PDFs)
- Processes them to create embeddings (vector representations)
- Allows users to query these papers with natural language questions
- Returns answers with citations and context from the relevant papers

Notes:

- MINE-DD uses Ollama models locally
- Default LLMs: ollama/llama3.2:1b (for laptop-friendly usage)
- Default embeddings: ollama/mxbai-embed-large:latest
- It uses Python >= 3.11

## Installation

### Requirements

- Python 3.11 or higher
- [Ollama](https://ollama.ai/) installed locally for running LLMs and embeddings

### Installation Steps

1. Clone the repository:

```console
git clone git@github.com:MINE-DD/MINE-DD.git
cd MINE-DD
```

2. Create a new virtual environment:

```console
python -m venv .venv
source .venv/bin/activate
```

2. Install the package:

```console
# Standard installation
python -m pip install .

# Development installation
python -m pip install -e ".[dev]"
```

### Testing

The project includes two types of tests:

1. **Standard Tests**: These run in CI environments (GitHub Actions) and don't require Ollama or GPU access.

   ```console
   # Run all standard tests
   pytest
   
   # Run specific test files
   pytest tests/test_utils.py tests/test_cli.py
   ```

2. **Integration Tests**: These test the full functionality including LLM queries with Ollama, requiring a local environment with Ollama running.

   ```console
   # Enable integration tests by setting SKIP_OLLAMA_TESTS=False in tests/test_query_integration.py
   # Then run:
   pytest -m integration
   ```

Integration tests are automatically skipped in CI environments and by default are also skipped locally (to avoid unexpected failures). To run them, you need to:

1. Ensure Ollama is running (`ollama serve`)
2. Set `SKIP_OLLAMA_TESTS = False` in `tests/test_query_integration.py`
3. Run with the integration marker: `pytest -m integration`


## Usage (PaperQA Version)

The main version of our system runs on top of [PaperQA](https://github.com/Future-House/paper-qa). This is an Agentic RAG+Evidence finding system with wonderful results, with the big caveat that it uses significant resources. Jump to the [Lightweight RAG version](#usage-lightweight-rag-version) is you prefer a faster solution that can be run entirely locally in your laptop (performance might be compromised so make sure to evaluate the quality of results for your use case).

### Before Using MINE-DD

Make sure Ollama is running in the background:

```console
ollama serve
```

Make sure to install the local LLMs you will want to use. The recommended models can be installed with:

```console
ollama pull mxbai-embed-large:latest
ollama pull llama3.2:latest
```

### Creating PaperQA Embeddings

Use the `minedd embed` command to create embeddings based on your document collection:

```console
minedd embed --paper_directory "/path/to/papers_minedd/" --embeddings_filename my-embeddings.pkl
```

#### Available Parameters

- `--embeddings_filename`: Name the embeddings pickle file where the index is saved
- `--output_dir`: Directory to save the embeddings pkl (default: 'out')
- `--embedding_model`: Embedding model (default: 'ollama/mxbai-embed-large:latest')
- `--paper_directory`: Directory with paper files (default: 'data/')
- `--augment_existing`: If True it will add new documents to the existing pkl file provided. Otherwise it creates the pkl file from scratch.


### Querying Papers

Use the `minedd query` command to ask questions about your document collection:

```console
minedd query --embeddings embeddings/papers_embeddings.pkl --questions_file questions.xlsx --output_dir results/
```

or for a single question:

```console
minedd query --embeddings embeddings/papers_embeddings.pkl --question "What is the relationship between climate change and diarrheal disease?"
```

#### Available Parameters

- `--embeddings`: Path to the embeddings pickle file (required)
- `--questions_file`: Path to Excel file with questions
- `--question`: Single question to ask
- `--llm`: LLM model to use (default: 'ollama/llama3.2:1b')
- `--embedding_model`: Embedding model (default: 'ollama/mxbai-embed-large:latest')
- `--paper_directory`: Directory with paper files (default: 'data/')
- `--output_dir`: Directory to save outputs (default: 'out')
- `--max_retries`: Retries for model loading failures (default: 2)

## Usage (Lightweight RAG Version)

Run this version for a non-agentic RAG system. With this version you: 
- Have much more control over the PDF pre-processing stages
- Can access document content in Markdown Format
- Can access document content in structured JSON Format
- Can use a local *Visual RAG* Interface with attribution in the generated answers.

There are three sequential stages for using the lightweight RAG:
1. Install software and declare relevant environment variables
2. Process PDFs (to obtain JSON files with parsed content) with `tools/extract_pdf_content.py`
3. Use the Visual RAG tool with the `tools/visual_rag.py` inside this same repository

### Install GROBID

This version uses [GROBID](https://github.com/kermitt2/grobid/tree/92ea31edc391c56f6fd1eab61a16aaab5fda960c) as the PDF preprocessor. It also makes use of [Marker](https://github.com/datalab-to/marker) to extract PDF images and convert content to Markdown. We also use [GMFT](https://github.com/conjuncts/gmft) to extract tables. Everything is included the MINE-DD package. Howver, the GROBID server is a Java (JVK >= 11 needed!) standalone application that must be installed separately with the following steps:

Open a separate terminal window to manage the GROBID Server.

For installing Docker-based GROBID (Recommended for Windows and Linux users):
1. Download [Docker Desktop](https://www.docker.com/products/docker-desktop/)
2. Pull the recommended lightweight image with `docker pull lfoppiano/grobid:0.8.1`
3. Run the image with `docker run -p 8070:8070 lfoppiano/grobid:0.8.2`

For installing GROBID from Source (Recommended for MAC users):
1. Download the source package: `wget https://github.com/kermitt2/grobid/archive/0.8.2.zip`
2. Go into Grobid Dir `cd grobid-0.8.2`
3. Install with `./gradlew clean install`
4. Run the server with `./gradlew run`

For more information about models,configuration and deployment visit the [Documentation](https://grobid.readthedocs.io/en/latest/Install-Grobid/) or this [GROBID Wiki](https://deepwiki.com/kermitt2/grobid/3-installation-and-deployment)

### RAG Environment Variables

Back in the terminal inside the MINE-DD repository you should create an environment file called `.env` in the root folder of the repository and manually declare the following variables with the desired values for you:

1. Create the file:

```console
vi .env
```

2. Open the file, copy the following content and modify to serve your purposes:

```python
PAPERS_DIRECTORY="/Path/to/papers_minedd" # Full Path to any PDF Paper Folder location
SAVE_VECTOR_INDEX="minedd_rag_index" # where the Vector Database is saved. Can be a full path or just a folder name
LLM_EMBEDDER="mxbai-embed-large:latest" # Embedding Model. We will ALWAYS embedd the documents locally
AVAILABLE_LLMS="ollama/llama3.2:latest,google_genai/gemini-2.5-flash-lite-preview-06-17" # A list of models, separated by commas. The syntax should be <provider>/<specific_model>, ...
MINEDD_EMBEDDINGS_PATH="minedd_paperqa_embeddings" # This is only needed if using visual_minedd.py (PaperQA version)
GOOGLE_API_KEY="copy-your-personal-api-key" # (only needed if using a Remote Gemini account). DO NOT SHARE THIS ONLINE!!!
```

### Process PDFs

The easiest is to use the Default values from the `.env` file. The script will search for every PDf inside the `PAPERS_DIRECTORY`, parse them and save a `.json` version for each processed pdf inside the same directory. You only need to directly run: 

```console
python tools/extract_pdf_content.py --mode all --skip_existing
``` 

If there are too many documents and you interrupt the script, next time you run it it will only process the pdf's which do not have a corresponding parsed JSON. Feel free to explore the script for more customizable parameters.

### Run Visual RAG

Once everything was setup you only need to run the following tool. You only need to change the `PAPERS_DIRECTORY` and `SAVE_VECTOR_INDEX` to generate different RAG systems pointing to different document collections. To run the visual RAg, make sure to be in the root of the repository and run:

```console
python -m streamlit run tools/visual_rag.py
``` 

### Creating Several Collections

As mentioned, you can manage separate paper collections as long as the collections are grouped inside the same directory and you point to the correct index. The cycle to create new collections is the following:
1. Put papers in a folder, e.g. `my_unique_papers`
2. Adjust the `PAPERS_DIRECTORY` and `SAVE_VECTOR_INDEX` variables in the `.env` file. Make sure the right index and the document folder always correspond to each other.
3. Run the pdf `extract_pdf_content.py` tool to create the jsons
4. Run Visual RAG `python -m streamlit run tools/visual_rag.py`



## Contributing

If you want to contribute to the development of MINE-DD,
have a look at the [contribution guidelines](CONTRIBUTING.md).

## License
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

## Credits

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [NLeSC/python-template](https://github.com/NLeSC/python-template).
