# Hybrid Language Identification using DistilBERT, FastText, and N-gram Filtering

_A Kaggle competition project by Oscar Pastural, Clément Florval, and Louis Gauthier_

This repository contains a complete pipeline for a language identification task that combines three components:
- **DistilBERT Classifier:** A fine-tuned multilingual DistilBERT model.
- **FastText Predictions:** Utilization of the [cis-lmu/glotlid](https://huggingface.co/cis-lmu/glotlid) FastText model.
- **N-gram Filtering:** A custom method using character-level n-grams (from unigrams to 4-grams) to refine predictions.

The project was developed during a Kaggle competition.

---

## Table of Contents

- [Hybrid Language Identification using DistilBERT, FastText, and N-gram Filtering](#hybrid-language-identification-using-distilbert-fasttext-and-n-gram-filtering)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Project Overview](#project-overview)
  - [Data and Exploratory Data Analysis (EDA)](#data-and-exploratory-data-analysis-eda)
  - [Modeling Pipeline](#modeling-pipeline)
    - [1. Data Preparation and Tokenization](#1-data-preparation-and-tokenization)
    - [2. N-gram Vocabulary and Filtering](#2-n-gram-vocabulary-and-filtering)
    - [3. Model Training and Evaluation](#3-model-training-and-evaluation)
    - [4. Ensemble and Final Prediction](#4-ensemble-and-final-prediction)
  - [Usage](#usage)

---

## Installation

Ensure you have Python 3.7+ installed. Install the necessary dependencies by running:

[CODE]bash
%pip install transformers datasets scikit-learn evaluate pandas matplotlib seaborn torch torchvision torchaudio "transformers[torch]" "git+https://github.com/thomas-ferraz/fastText.git@0ffafb256d3f6325f8e5dbd527b4e22c1a7e49d2"
[/CODE]

This command installs:
- **Transformers & Datasets:** For tokenization and model loading.
- **scikit-learn & evaluate:** For preprocessing and metric evaluation.
- **PyTorch:** For model training and inference.
- **FastText:** For alternative language identification.

---

## Project Overview

This project aims to improve automatic language identification by leveraging a hybrid approach that integrates:
- A **DistilBERT-based classifier** fine-tuned on multilingual text.
- **FastText predictions** to supplement and cross-check language labels.
- A **novel n-gram filtering mechanism** that calculates the match between text n-grams and pre-built language vocabularies.

The end-to-end pipeline includes data loading, exploratory data analysis (EDA), model training, prediction filtering, ensemble methods, and final export of results.

---

## Data and Exploratory Data Analysis (EDA)

The repository expects two CSV files:
- `data/train_submission.csv`: Training data with text and language labels.
- `data/test_without_labels.csv`: Test data without labels.

In the EDA section:
- Basic information about dataset dimensions, missing values, and unique language labels is printed.
- Visualization of the label distribution and text length distributions is performed using Matplotlib and Seaborn.
- Examples of texts per language are displayed to assess data quality.

---

## Modeling Pipeline

### 1. Data Preparation and Tokenization

- **Loading Data:** The CSV files are loaded using Pandas.
- **Preprocessing:** Missing values in the text column are filled and converted to strings.
- **Label Encoding:** The language labels are encoded using `LabelEncoder`.
- **Conversion to Hugging Face Dataset:** DataFrames are converted to a Dataset object.
- **Tokenization:** A pre-trained tokenizer (from `distilbert-base-multilingual-cased`) tokenizes the text with padding and truncation.

### 2. N-gram Vocabulary and Filtering

- **Building n-gram Vocabularies:** For each language, n-gram vocabularies (n=1 to 4) are constructed from training texts.
- **Filtering Function:** A function computes the fraction of n-grams in a given text that match the pre-built vocabulary. If the fraction is below a specified threshold, the candidate language is filtered out.

### 3. Model Training and Evaluation

- **Training DistilBERT:** The DistilBERT model is fine-tuned with training arguments (e.g., 5 epochs, batch size 16, learning rate 2e-5).
- **Evaluation:** Accuracy is calculated on a held-out validation set. Predictions are made on both evaluation and test splits.
- **Saving Model and Predictions:** Trained model weights and predictions are saved for later use.

### 4. Ensemble and Final Prediction

- **FastText Predictions:** The FastText model (`cis-lmu/glotlid`) is used to predict languages on evaluation and test sets.
- **Combined Model:** Predictions from DistilBERT (filtered via n-gram thresholds) and FastText are combined. A group-based decision strategy selects the final label based on historical accuracy metrics.
- **Exporting Results:** Final predictions are exported to CSV files, ready for submission.

---

## Usage

1. **Installation:** Follow the installation instructions above.
2. **Data Preparation:** Place your training and test CSV files in the `data/` directory.
3. **Running the Notebook/Scripts:**
   - Execute the provided Jupyter notebook or run the Python scripts sequentially to reproduce data loading, training, evaluation, and prediction export.
4. **Compilation (if applicable):** The LaTeX report is not included in this repository. Use the provided scripts and notebooks to generate results and predictions.
