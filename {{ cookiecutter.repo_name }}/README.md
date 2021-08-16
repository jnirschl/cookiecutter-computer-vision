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

## Data science life cycle

### 1. Domain understanding/problem definition

The first step in any data science life cycle is to define the question and to understand the problem domain and prior
knowledge. Given a well-formulated question, the team can specify the goal of the machine learning application (e.g.,
regression, classification, clustering, outlier detection) and how it will be measured, and which data sources will be
needed. The scope of the project, key personnel, key milestones, and general project architecture/overview is specified
in the Project Charter and iterated throughout the life of the project. A list of data sources which are available or
need to be collected is specified in the table of data sources. Finally, the existing data is summarized in a data
dictionary that describes the features, number of elements, non-null data, data type (e.g., nominal, ordinal,
continuous), data range, as well as a table with key descriptive summary statistics.

*Deliverables Step 1:*
1. Project charter
2. Table of data sources
3. Data dictionary
4. Summary table of raw dataset

#### Downloading the dataset

The script [make_dataset.py](src/data/make_dataset.py) will download the dataset from Kaggle, create a data dictionary,
and summarize the dataset using [TableOne](https://pypi.org/project/tableone/). The key artifacts of this DVC stage are
the [raw training and testing datasets](data/raw), the [data_dictionary](reports/figures/data_dictionary.tex), and
the [summary table](/reports/figures/table_one.tex).

In your terminal, use the command-line interface to build the first stage of the pipeline.

``` bash
dvc run -n make_dataset -p color_mode,target_size \
-d src/data/make_dataset.py \
-d data/raw \
-o data/interim/label_encoding.yaml \
-o data/interim/mapfile_df.csv \
-o data/interim/mean_image.png \
-o data/processed/split_train_dev.csv \
--desc "Data processing script to create a mapfile from the specified data directory, split into train/dev/test, and compute the mean image"\
 python3 src/data/make_dataset.py "data/raw" "data/processed" "mapfile_df.csv" --force
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
