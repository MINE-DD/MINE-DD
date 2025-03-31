[![github repo badge](https://img.shields.io/badge/github-repo-000.svg?logo=github&labelColor=gray&color=blue)](https://github.com/MINE-DD/mine-dd)
[![github license badge](https://img.shields.io/github/license/MINE-DD/mine-dd)](https://github.com/MINE-DD/mine-dd )
[![RSD](https://img.shields.io/badge/rsd-mine_dd-00a3e3.svg)](https://research-software-directory.org/projects/mine-dd)
[![fair-software badge](https://img.shields.io/badge/fair--software.eu-%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8B-yellow)](https://fair-software.eu)
[![workflow scq badge](https://sonarcloud.io/api/project_badges/measure?project=MINE-DD_mine-dd&metric=alert_status)](https://sonarcloud.io/dashboard?id=MINE-DD_MINE-DD)
[![workflow scc badge](https://sonarcloud.io/api/project_badges/measure?project=MINE-DD_mine-dd&metric=coverage)](https://sonarcloud.io/dashboard?id=MINE-DD_MINE-DD)
[![build](https://github.com/MINE-DD/mine-dd/actions/workflows/build.yml/badge.svg)](https://github.com/MINE-DD/mine-dd/actions/workflows/build.yml)
[![cffconvert](https://github.com/MINE-DD/mine-dd/actions/workflows/cffconvert.yml/badge.svg)](https://github.com/MINE-DD/mine-dd/actions/workflows/cffconvert.yml)
[![sonarcloud](https://github.com/MINE-DD/mine-dd/actions/workflows/sonarcloud.yml/badge.svg)](https://github.com/MINE-DD/mine-dd/actions/workflows/sonarcloud.yml)
[![markdown-link-check](https://github.com/MINE-DD/mine-dd/actions/workflows/markdown-link-check.yml/badge.svg)](https://github.com/MINE-DD/mine-dd/actions/workflows/markdown-link-check.yml)

## MINE-DD

Mining the past to protect against diarrheal disease in the future.

## Description

MINE-DD is a Python package that uses artificial intelligence to mine existing literature to describe how climate might change diarrheal diseases in the future. The package helps project the risk of diarrheal disease into the future, preparing communities and informing policy makers.

## Installation

To install MINE-DD from GitHub repository, do:

```console
git clone git@github.com:MINE-DD/MINE-DD.git
cd MINE-DD
python -m pip install .
```

## Usage

Creating embeddings from papers:

```console
minedd-embed --papers-dir papers/ --metadata metadata_papers.csv --output-dir embeddings/
```

Querying the papers:

```console
minedd-query --embeddings embeddings/papers_embeddings.pkl --questions questions.xlsx --output-dir results/
```

## Contributing

If you want to contribute to the development of mine_dd,
have a look at the [contribution guidelines](CONTRIBUTING.md).

## License
Apache License 2.0

## Credits

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [NLeSC/python-template](https://github.com/NLeSC/python-template).
