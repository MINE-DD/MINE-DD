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


## Usage

### Before Using MINE-DD

Make sure Ollama is running in the background:

```console
ollama serve
```

### Creating Paper Embeddings

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

## Contributing

If you want to contribute to the development of MINE-DD,
have a look at the [contribution guidelines](CONTRIBUTING.md).

## License
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

## Credits

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [NLeSC/python-template](https://github.com/NLeSC/python-template).
