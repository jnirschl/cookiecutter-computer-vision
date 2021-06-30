# {{cookiecutter.project_name}}
![GitHub](https://img.shields.io/github/license/{{cookiecutter.github_username}}/{{cookiecutter.repo_name}})
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
<!---
Add Zenodo DOI after first release
[![DOI](https://zenodo.org/badge/123456789.svg)](https://zenodo.org/badge/latestdoi/123456789)
--->

==============================

## Project goals
{{cookiecutter.description}}

## Contents
* [Getting started](#getting-started)
* [Project organization](#-project-organization)

## Getting started

This repository uses [Data Version Control (DVC)](https://dvc.org/) to create a
machine learning pipeline and track experiments. We will use a modified version
of the [Team Data Science Process](https://docs.microsoft.com/en-us/azure/machine-learning/team-data-science-process/overview)
as our Data Science Life cycle template. This repository template is based on the
[cookiecutter data-science project template](https://drivendata.github.io/cookiecutter-data-science).

In order to start, clone this repository and install [DataVersionControl](https://dvc.org/).
Next, pip install the requirements according to the script below. Finally, pull
the latest version of data and trained models, which are hosted on
[DagsHub](https://dagshub.com/{{cookiecutter.github_username}}/{{cookiecutter.repo_name}}).

```bash
# clone the repository
git clone https://github.com/{{ cookiecutter.github_username }}/{{ cookiecutter.repo_name }}.git

# create virtual environment in folder
cd {{ cookiecutter.repo_name }}
python3 -m venv venv
source venv/bin/activate

# install requirements
pip3 install -r requirements.txt
pip3 install .

# pull data from origin (https://dagshub.com/{{ cookiecutter.github_username }}/{{ cookiecutter.repo_name }})
dvc pull -r origin

# check the status of the pipleline
dvc status

# Expected output
#   Data and pipelines are up to date.
```


## Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── pytest.ini         <- ini file with settings for running pytest; see pytest.readthedocs.io
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecuttercomputervision</small></p>
