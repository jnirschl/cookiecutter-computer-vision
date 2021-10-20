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
dvc init
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
in the [Project Charter](#reports/project/charter.md) and iterated throughout the life of the project. A list of data sources which are available or
need to be collected is specified in the table of data sources. Finally, the existing data is summarized in a data
dictionary that describes the features, number of elements, non-null data, data type (e.g., nominal, ordinal,
continuous), data range, as well as a table with key descriptive summary statistics.

*Deliverables Step 1:*
1. [Project charter](#reports/project/charter.md)
2. Table of data sources
3. [Data dictionary](#reports/data_dictionaries/)
4. Summary table of raw dataset

#### Preparing the dataset

The script [make_dataset.py](src/data/make_dataset.py) will download the dataset from Kaggle, create a data dictionary,
and summarize the dataset using [TableOne](https://pypi.org/project/tableone/). The key artifacts of this DVC stage are
the [raw training and testing datasets](data/raw), the [data_dictionary](reports/figures/data_dictionary.tex), and
the [summary table](/reports/figures/table_one.tex).

In your terminal, use the command-line interface to add the raw immutable data.

``` bash
dvc add data/raw/data data/raw/mask --desc "Add raw data to version control"

# add .dvc file to version control 
git add -f data/raw/data.dvc data/raw/mask.dvc
```

Now, create a stage to break images into smaller tiles using a custom shell script.
``` bash
dvc run -f -n split_images \
-d src/data/split_images.sh \
-d data/raw/ \
-o data/processed/data \
-o data/processed/mask \
--desc "Shell script to split images into 256x256 tiles and save in data/processed/"\
bash src/data/split_images.sh

git add dvc.yaml dvc.lock
```

### Nerve morphometry
The raw data has already been added to version control and large images have been split  
into 256x256 tiles. The next stage will create the mapfile, compute the mean and std 
images, and create train/test splits.
``` bash
dvc run -n make_dataset -p color_mode,save_format,segmentation,target_size \
-d src/data/make_dataset.py \
-d data/processed/data \
-d data/processed/mask \
-o data/processed/mapfile_df.csv \
-o data/processed/split_train_dev.csv \
-o data/processed/mean_image.png \
-o data/processed/std_image.png \
--desc "Create a mapfile from the directories, compute the mean and std image, and split into train/dev/test sets." \
python3 src/data/make_dataset.py data/processed/ data/processed/ mapfile_df.csv -p params.yaml --force
```

We will use the mapfile and train/dev splits to train a model in the next stage.
``` bash
dvc run -n train_seg -p color_mode,mean_img,std_img,train_model,random_seed,segmentation,target_size,n_classes \
-d src/models/train_model.py \
-d data/processed/mapfile_df.csv \
-d data/processed/split_train_dev.csv \
-d data/processed/mean_image.png \
-d data/processed/std_image.png \
-o models/ \
-m results/metrics.json \
--desc "Train a model using the mapfile and train/dev splits." \
python3 src/models/train_model.py data/processed/mapfile_df.csv data/processed/split_train_dev.csv -p params.yaml --model-name nerve_seg
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
