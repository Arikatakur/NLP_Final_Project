# NLP Final Project

Text classification final project for the NLP course.

## Current Scope

Part 1 is implemented:

- load review data into `pandas`
- inspect dataset size, review lengths, and label balance
- create a category distribution graph
- train a Bag-of-Words baseline classifier
- report baseline accuracy and F1

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run Part 1

Smoke test with the included small sample dataset:

```bash
python -m src.nlp_final_project.part1_baseline --csv-path data/sample_reviews.csv
```

Run with the real IMDB dataset after placing it at `data/imdb_reviews.csv`:

```bash
python -m src.nlp_final_project.part1_baseline --csv-path data/imdb_reviews.csv --sample-size 5000
```

Expected CSV columns:

- review text column: `review`, `text`, `content`, or `sentence`
- label column: `sentiment`, `label`, `target`, or `category`

Supported labels include `positive`/`negative`, `pos`/`neg`, and `1`/`0`.

## Outputs

Part 1 writes results under `outputs/part1/`:

- `category_distribution.png`
- `dataset_summary.csv`
- `metrics.json`
- `classification_report.txt`
