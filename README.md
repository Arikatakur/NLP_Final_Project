# NLP Final Project - Sentiment Classification

Text classification final project for the Natural Language Processing course.

## Submitters

- Saleem Yousef
- Noor Shama

## What This Project Does

This project classifies English IMDB movie reviews as positive or negative. It starts with a Bag-of-Words baseline, adds preprocessing, compares several classical ML classifiers with cross-validation, tunes and analyzes the best classical model, and compares the result against a pretrained HuggingFace sentiment model.

## Current Status

The code is implemented and real results were generated from `data/imdb_reviews.csv` using a balanced 5,000-review subset.

Final real-results summary:

| Stage | Best / Main Approach | Accuracy | Macro F1 |
|---|---|---:|---:|
| Part 1 | Bag-of-Words + Logistic Regression | 0.851 | 0.851 |
| Part 2 | Preprocessed Bag-of-Words + Logistic Regression | 0.836 | 0.836 |
| Part 3 | TF-IDF + Logistic Regression, 5-fold CV | 0.866 | 0.866 |
| Part 4 | TF-IDF bigrams + tuned Logistic Regression, 5-fold CV | 0.867 | 0.867 |
| Part 5 | HuggingFace DistilBERT sentiment pipeline | 0.878 | 0.878 |

The final written 2-3 page report still needs to be prepared before Moodle submission. Use `docs/FINAL_REPORT_TEMPLATE.md` and the generated `outputs/` files.

## Repository Structure

```text
data/
  sample_reviews.csv              # tracked smoke-test data
  imdb_reviews.csv                # local real dataset, ignored by Git
src/nlp_final_project/
  data.py                         # CSV loading and stratified sampling
  text_processing.py              # preprocessing function
  features.py                     # review-length feature transformer
  modeling.py                     # shared model builders
  part1_baseline.py
  part2_preprocessing.py
  part3_classifier_comparison.py
  part4_improve_analyze.py
  part5_pretrained_compare.py
docs/
  ASSIGNMENT_REQUIREMENTS.md
  BEFORE_SUBMISSION.md
  FINAL_REPORT_TEMPLATE.md
  PROJECT_GUIDE.md
  REAL_RESULTS_SUMMARY.md
outputs/                          # generated local results, ignored by Git
requirements.txt
requirements-pretrained.txt
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

For the HuggingFace pretrained model:

```bash
pip install -r requirements-pretrained.txt
```

## Dataset

The real dataset should be placed at:

```text
data/imdb_reviews.csv
```

Expected columns:

- text column: `review`, `text`, `content`, or `sentence`
- label column: `sentiment`, `label`, `target`, or `category`

Accepted labels:

- positive: `positive`, `pos`, `1`, `true`
- negative: `negative`, `neg`, `0`, `false`

## Run The Project

Smoke test with the included sample data:

```bash
python -m src.nlp_final_project.part1_baseline --csv-path data/sample_reviews.csv
python -m src.nlp_final_project.part2_preprocessing --csv-path data/sample_reviews.csv
python -m src.nlp_final_project.part3_classifier_comparison --csv-path data/sample_reviews.csv
python -m src.nlp_final_project.part4_improve_analyze --csv-path data/sample_reviews.csv
python -m src.nlp_final_project.part5_pretrained_compare --csv-path data/sample_reviews.csv --skip-pretrained
```

Run on the real IMDB dataset:

```bash
python -m src.nlp_final_project.part1_baseline --csv-path data/imdb_reviews.csv --sample-size 5000
python -m src.nlp_final_project.part2_preprocessing --csv-path data/imdb_reviews.csv --sample-size 5000
python -m src.nlp_final_project.part3_classifier_comparison --csv-path data/imdb_reviews.csv --sample-size 5000
python -m src.nlp_final_project.part4_improve_analyze --csv-path data/imdb_reviews.csv --sample-size 5000
python -m src.nlp_final_project.part5_pretrained_compare --csv-path data/imdb_reviews.csv --sample-size 5000 --max-pretrained-samples 500
```

Part 5 uses original raw text for the pretrained model, not preprocessed/stemmed text.

## Generated Outputs

Generated outputs are written under `outputs/` and ignored by Git.

Important submission files:

- `outputs/part1/category_distribution.png`
- `outputs/part1/dataset_summary.csv`
- `outputs/part1/metrics.json`
- `outputs/part2/preprocessing_comparison.csv`
- `outputs/part2/conclusion.txt`
- `outputs/part3/classifier_comparison.csv`
- `outputs/part3/best_confusion_matrix.png`
- `outputs/part4/improvement_table.csv`
- `outputs/part4/influential_words.csv`
- `outputs/part4/error_analysis.csv`
- `outputs/part5/final_comparison.csv`
- `outputs/part5/part4_error_recheck.csv`

## Documentation

- `docs/PROJECT_GUIDE.md`: detailed implementation and run guide
- `docs/REAL_RESULTS_SUMMARY.md`: real result summary for the report
- `docs/FINAL_REPORT_TEMPLATE.md`: 2-3 page report outline
- `docs/BEFORE_SUBMISSION.md`: final checklist
- `docs/ASSIGNMENT_REQUIREMENTS.md`: extracted assignment requirements

## Submission Checklist

Before final submission:

- write the 2-3 page report;
- include generated tables/graphs as appendix material;
- review the five errors in `outputs/part4/error_analysis.csv`;
- confirm whether Moodle expects a GitHub link, ZIP file, PDF report, or all of them;
- make sure both submitters can explain every part for the defense.