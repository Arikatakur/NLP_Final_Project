# Real Results Summary

These results were generated from `data/imdb_reviews.csv` using a balanced 5,000-review subset of the IMDB dataset.

## Dataset

- Source file: `data/imdb_reviews.csv`
- Columns: `review`, `sentiment`
- Working subset: 5,000 reviews
- Class balance: 2,500 positive and 2,500 negative reviews
- Random state: 42

## Main Results

| Stage | Model / Approach | Evaluation | Accuracy | Macro F1 |
|---|---|---:|---:|---:|
| Part 1 | Bag-of-Words + Logistic Regression | 1,000-review holdout | 0.851 | 0.851 |
| Part 2 | Preprocessed Bag-of-Words + Logistic Regression | 1,000-review holdout | 0.836 | 0.836 |
| Part 3 | TF-IDF + Logistic Regression | 5-fold CV | 0.866 | 0.866 |
| Part 4 | TF-IDF bigrams + tuned Logistic Regression `C=2.0` | 5-fold CV | 0.867 | 0.867 |
| Part 5 | HuggingFace DistilBERT sentiment pipeline | 500-review evaluation subset | 0.878 | 0.878 |

## Interpretation

The simple Bag-of-Words baseline was already strong. Preprocessing reduced performance slightly, probably because stemming and stop-word removal removed useful sentiment signals and phrase context. TF-IDF with bigrams improved the classical approach, and the pretrained HuggingFace model performed best overall because it can use word order and contextual information that classical sparse-feature models mostly ignore.

## Output Files To Use In The Report

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