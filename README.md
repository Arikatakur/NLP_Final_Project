# NLP Final Project

Text classification final project for the NLP course.

## Implemented Parts

- Part 1: data loading, dataset exploration, category distribution graph, and Bag-of-Words baseline.
- Part 2: preprocessing with tokenization, normalization, stop-word removal, stemming, and before/after comparison.
- Part 3: TF-IDF plus review-length features, cross-validation, classifier comparison, and confusion matrix.
- Part 4: measured tuning variants, influential-word extraction, and error analysis.
- Part 5: baseline/classical/pretrained comparison with HuggingFace support for raw text.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Optional dependency set for running the HuggingFace pretrained model locally:

```bash
pip install -r requirements-pretrained.txt
```

## Dataset

Place the real dataset at `data/imdb_reviews.csv` or pass another path with `--csv-path`.

Expected CSV columns:

- review text column: `review`, `text`, `content`, or `sentence`
- label column: `sentiment`, `label`, `target`, or `category`

Supported labels include `positive`/`negative`, `pos`/`neg`, and `1`/`0`.

For IMDB, use a stable subset of 3,000-5,000 reviews across all parts:

```bash
--sample-size 5000
```

## Run Commands

Smoke test with the included sample dataset:

```bash
python -m src.nlp_final_project.part1_baseline --csv-path data/sample_reviews.csv
python -m src.nlp_final_project.part2_preprocessing --csv-path data/sample_reviews.csv
python -m src.nlp_final_project.part3_classifier_comparison --csv-path data/sample_reviews.csv
python -m src.nlp_final_project.part4_improve_analyze --csv-path data/sample_reviews.csv
python -m src.nlp_final_project.part5_pretrained_compare --csv-path data/sample_reviews.csv --skip-pretrained
```

Run the full workflow on the real IMDB subset:

```bash
python -m src.nlp_final_project.part1_baseline --csv-path data/imdb_reviews.csv --sample-size 5000
python -m src.nlp_final_project.part2_preprocessing --csv-path data/imdb_reviews.csv --sample-size 5000
python -m src.nlp_final_project.part3_classifier_comparison --csv-path data/imdb_reviews.csv --sample-size 5000
python -m src.nlp_final_project.part4_improve_analyze --csv-path data/imdb_reviews.csv --sample-size 5000
python -m src.nlp_final_project.part5_pretrained_compare --csv-path data/imdb_reviews.csv --sample-size 5000 --max-pretrained-samples 500
```

Part 5 uses original raw text for HuggingFace, not preprocessed/stemmed text.

## Outputs

Generated outputs are written under `outputs/` and ignored by Git.

- Part 1: `outputs/part1/category_distribution.png`, `dataset_summary.csv`, `metrics.json`, `classification_report.txt`
- Part 2: `outputs/part2/preprocessing_comparison.csv`, `classification_reports.json`, `conclusion.txt`
- Part 3: `outputs/part3/classifier_comparison.csv`, `best_confusion_matrix.csv`, `best_confusion_matrix.png`, `part3_summary.json`
- Part 4: `outputs/part4/improvement_table.csv`, `influential_words.csv`, `error_analysis.csv`, `part4_summary.json`
- Part 5: `outputs/part5/final_comparison.csv`, `part4_error_recheck.csv`, `part5_summary.json`