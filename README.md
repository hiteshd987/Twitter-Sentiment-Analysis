# Twitter sentiment analysis Project

This project performs sentiment analysis on Twitter data using the Sentiment140 dataset from Kaggle and uses different machine learning models with the lexicon tools as a baseline for performance comparison.

## Folder structure

- `utils/`: Contains dataset loader file.
- `models/`: Contains Python files for different machine learning models.
- `preprocessing/`: Contains script for data preprocessing.

## Dataset CSV

Install the sentiment140 dataset from Kaggle (https://www.kaggle.com/datasets/kazanova/sentiment140) 

Create a folder as 'data' and save the downloaded file with the 'twitterDataset.csv' name

## Install dependencies

You can install the required dependencies using pip:

!pip install pandas nltk matplotlib scikit-learn tensorflow gensim numpy emoji

## Implementation

Run sentiment_analysis.ipynb file and each methods will get called starting from loaddataset, preprocessing to models.

