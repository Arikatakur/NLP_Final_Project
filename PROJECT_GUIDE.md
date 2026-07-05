# Project Guide

## What Was Implemented

This project implements the full NLP final project as five runnable stages for binary sentiment classification.

Part 1 builds the first baseline. It loads review data with `pandas`, summarizes the dataset, creates a category distribution graph, vectorizes reviews with Bag-of-Words, trains a simple classifier, and writes baseline accuracy/F1 outputs.

Part 2 adds text preprocessing. It normalizes text, tokenizes it, removes English stop words, applies stemming, and compares the same Bag-of-Words baseline before and after preprocessing.

Part 3 upgrades the feature and model comparison stage. It uses TF-IDF, optional bigrams, review-length features, cross-validation, and compares Naive Bayes, Logistic Regression, SVM, kNN, and Random Forest as the additional independently researched algorithm.

Part 4 improves and analyzes the best classical approach. It runs measured tuning variants, extracts influential words from the linear model, writes an improvement table, and produces an error-analysis file for misclassified examples.

Part 5 compares the classical workflow against a pretrained HuggingFace sentiment model. The pretrained model receives raw original text, not preprocessed or stemmed text. The script also supports `--skip-pretrained` for machines where HuggingFace dependencies or model files are unavailable.

## Project Structure

```text
data/
  sample_reviews.csv
src/nlp_final_project/
  data.py
  features.py
  modeling.py
  text_processing.py
  part1_baseline.py
  part2_preprocessing.py
  part3_classifier_comparison.py
  part4_improve_analyze.py
  part5_pretrained_compare.py
requirements.txt
requirements-pretrained.txt
README.md
requirements.md
```

Generated files are written to `outputs/` and are ignored by Git.

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install the core dependencies:

```bash
pip install -r requirements.txt
```

Install optional pretrained-model dependencies only if you want to run HuggingFace locally:

```bash
pip install -r requirements-pretrained.txt
```

## Dataset

The scripts expect a CSV file with one text column and one label column.

Accepted review text column names:

- `review`
- `text`
- `content`
- `sentence`

Accepted label column names:

- `sentiment`
- `label`
- `target`
- `category`

Accepted labels:

- positive: `positive`, `pos`, `1`, `true`
- negative: `negative`, `neg`, `0`, `false`

For the real project run, put the IMDB dataset at:

```text
data/imdb_reviews.csv
```

Use the same stable subset size for all parts, for example:

```bash
--sample-size 5000
```

## How To Run

Run all parts on the included small sample dataset:

```bash
python -m src.nlp_final_project.part1_baseline --csv-path data/sample_reviews.csv
python -m src.nlp_final_project.part2_preprocessing --csv-path data/sample_reviews.csv
python -m src.nlp_final_project.part3_classifier_comparison --csv-path data/sample_reviews.csv
python -m src.nlp_final_project.part4_improve_analyze --csv-path data/sample_reviews.csv
python -m src.nlp_final_project.part5_pretrained_compare --csv-path data/sample_reviews.csv --skip-pretrained
```

Run all parts on the real IMDB subset:

```bash
python -m src.nlp_final_project.part1_baseline --csv-path data/imdb_reviews.csv --sample-size 5000
python -m src.nlp_final_project.part2_preprocessing --csv-path data/imdb_reviews.csv --sample-size 5000
python -m src.nlp_final_project.part3_classifier_comparison --csv-path data/imdb_reviews.csv --sample-size 5000
python -m src.nlp_final_project.part4_improve_analyze --csv-path data/imdb_reviews.csv --sample-size 5000
python -m src.nlp_final_project.part5_pretrained_compare --csv-path data/imdb_reviews.csv --sample-size 5000 --max-pretrained-samples 500
```

If HuggingFace cannot run locally, use:

```bash
python -m src.nlp_final_project.part5_pretrained_compare --csv-path data/imdb_reviews.csv --sample-size 5000 --skip-pretrained
```

## Outputs

Part 1 writes:

- `outputs/part1/category_distribution.png`
- `outputs/part1/dataset_summary.csv`
- `outputs/part1/metrics.json`
- `outputs/part1/classification_report.txt`

Part 2 writes:

- `outputs/part2/preprocessing_comparison.csv`
- `outputs/part2/classification_reports.json`
- `outputs/part2/conclusion.txt`

Part 3 writes:

- `outputs/part3/classifier_comparison.csv`
- `outputs/part3/best_confusion_matrix.csv`
- `outputs/part3/best_confusion_matrix.png`
- `outputs/part3/part3_summary.json`

Part 4 writes:

- `outputs/part4/improvement_table.csv`
- `outputs/part4/influential_words.csv`
- `outputs/part4/error_analysis.csv`
- `outputs/part4/part4_summary.json`

Part 5 writes:

- `outputs/part5/final_comparison.csv`
- `outputs/part5/part4_error_recheck.csv`
- `outputs/part5/part5_summary.json`

## Notes

The included `data/sample_reviews.csv` is only for smoke testing. Its metrics are not meaningful because it contains very few rows. Use the real IMDB subset for report results.

Part 3 and Part 4 use cross-validation as required by the assignment.

Random Forest is included as the additional ML algorithm that was not part of the core required classifier list.