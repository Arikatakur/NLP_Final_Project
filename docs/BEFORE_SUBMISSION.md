# Before Submission Checklist

## 1. Add The Real Dataset

Place the real IMDB dataset here:

```text
data/imdb_reviews.csv
```

The CSV should include:

- review text column: `review`, `text`, `content`, or `sentence`
- label column: `sentiment`, `label`, `target`, or `category`
- labels such as `positive` / `negative`

Use the same stable subset size for every part, for example `5000` reviews.

## 2. Run All Parts On The Real Dataset

From the project root, run:

```bash
python -m src.nlp_final_project.part1_baseline --csv-path data/imdb_reviews.csv --sample-size 5000
python -m src.nlp_final_project.part2_preprocessing --csv-path data/imdb_reviews.csv --sample-size 5000
python -m src.nlp_final_project.part3_classifier_comparison --csv-path data/imdb_reviews.csv --sample-size 5000
python -m src.nlp_final_project.part4_improve_analyze --csv-path data/imdb_reviews.csv --sample-size 5000
python -m src.nlp_final_project.part5_pretrained_compare --csv-path data/imdb_reviews.csv --sample-size 5000 --max-pretrained-samples 500
```

If HuggingFace cannot run locally, run Part 5 in Google Colab or use:

```bash
python -m src.nlp_final_project.part5_pretrained_compare --csv-path data/imdb_reviews.csv --sample-size 5000 --skip-pretrained
```

For the best submission, include real pretrained-model results, not only `--skip-pretrained`.

## 3. Check The Generated Outputs

Make sure these files exist after running:

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

## 4. Confirm Assignment Requirements

Verify:

- Part 3 and Part 4 use cross-validation.
- Random Forest is included as the extra ML algorithm.
- Part 5 uses original raw text, not preprocessed/stemmed text.
- The final comparison table includes Part 1 through Part 5.
- Part 4 includes five analyzed errors. If fewer than five are found, add manually written challenge examples and analyze them.

## 5. Write The Final Report

Create a short report of 2-3 pages of text.

Recommended structure:

1. Project goal and dataset used.
2. Part 1 baseline result.
3. Effect of preprocessing from Part 2.
4. Classifier comparison from Part 3.
5. Improvements and error analysis from Part 4.
6. Pretrained model comparison from Part 5.
7. Final conclusion: which approach won and what the tradeoff was.

Put tables and graphs in an appendix so they do not count toward the 2-3 page text limit.

## 6. Prepare For The Defense

Both submitters should be able to explain:

- how the dataset is loaded and sampled;
- how Bag-of-Words and TF-IDF work;
- what preprocessing does;
- why cross-validation is used from Part 3 onward;
- how each classifier works at a high level;
- why Random Forest was chosen as the extra algorithm;
- what the confusion matrix shows;
- what the influential words mean;
- what the five analyzed errors reveal;
- why HuggingFace uses raw text;
- the tradeoff between classical ML and pretrained models.

## 7. Final Submission Package

Submit:

- the code project or GitHub link;
- the final 2-3 page report;
- generated outputs/tables/graphs as appendix material;
- any real dataset instructions if the dataset itself is too large to upload.

Do not rely on the included `data/sample_reviews.csv` for final results. It is only a smoke-test file.
