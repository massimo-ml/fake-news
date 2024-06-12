# fake-news

# Goal of the project

The main goal of this project is to determine whether the addition of synthetic data to the training set may
improve the predictive capabilities of fake-news detectors. 

# Methodology

We performed the following steps in our experiment:
1. Train a set of classifiers on `WELFake_Dataset` that contains about 56000 examples of titles and texts of articles. An article may either be a fake-news or a true article
2. Fine-tune a set of LLMs on the train data for classifiers
3. Evaluate the fine-tuned LLMs with the previously trained classifiers. 
4. Generate synthetic data with LLMs. Replace the longest articles in the dataset with generated, shorter articles (The generation is conditioned on the title of the article)
5. Train a new set of classifiers on the dataset extended with synthetic articles and compare the results

# Contents

- fake_news
    - classifiers - contains code for different classifiers
    - generators - contains code for different LLM-based generators
    - `base.py` - contains the definitions of abstract classes for generators and classifiers
    - `data_preprocessing.py` - contains code with input transformation for classifiers
    - `generator_evaluation.py` - contains a function that evaluates LLM-based generators with a set of previously trained classifiers
    - `metrics.py` - contains code responsible for calculating all necessary metrics for classifier evaluation
- notebooks - contains all jupyter notebooks
    - `classifiers.ipynb` - contains code for training the first set of classifiers (i.e. the ones that do not use the synthetic data)
    - `data_preparation.ipynb` - contains different statisics of the data
    - `generators_evaluation.ipynb` - contains evaluation of all of the generators
- scripts - contains different scripts to transform the data
    - `data_prep.py` - contains code for data cleaning and preparation
- test - contains unit tests

# Summary of the results

TODO

# Development

## Dataset
1. Download dataset from https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification
2. Extract and put CSV into data: `./data/WELFake_Dataset.csv`