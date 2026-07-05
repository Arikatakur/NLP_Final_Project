# Natural Language Processing - Final Project: Text Classification

## Source

- Local assignment HTML: `../nlp_final_project.html`
- Original Moodle URL: https://moodle.kinneret.ac.il/pluginfile.php/86596/mod_resource/content/1/nlp_final_project.html
- Course: Natural Language Processing / NLP
- Project: Final Project - Text Classification

## Goal

Build a text classifier that receives an English review and predicts whether it is positive or negative.

The project starts with a simple Bag-of-Words baseline, improves the preprocessing and feature representation step by step, compares multiple classical ML classifiers, analyzes mistakes, and finally compares the best classical approach against a pretrained HuggingFace sentiment model.

## General Requirements

- Submit in pairs or individually.
- After submission there will be a defense meeting.
- Every team member must be able to explain every part of the code, not only the part they wrote.
- Work in PyCharm with `.py` files or in a Jupyter Notebook.
- Text data is in English.
- Build one clean, documented project covering all five parts.
- Keep a results table from the beginning: each model/version, accuracy, F1, and notes.
- Change one thing at a time and measure the effect separately.
- Use a fixed `random_state` where relevant.

## Mandatory Requirements

1. Cross-validation:
   - Part 1 may use a simple train/test split as an initial baseline.
   - From Part 3 onward, every performance evaluation must use cross-validation, such as k-fold.
   - Missing cross-validation affects grading in Parts 3 and 4.

2. Additional ML algorithm:
   - Add one ML algorithm that was not taught in class.
   - Examples: Random Forest, Gradient Boosting, XGBoost.
   - Research it independently, run it on the task, and include it in the comparison table.
   - This is mandatory and counted in Part 3.

## Part 1 - Data and Baseline

Goal: understand the dataset and build the simplest classifier baseline.

Tasks:

- Load the reviews into `pandas`.
- Explore the data: number of examples, review lengths, and category balance.
- Use a simple train/test split for this first baseline only.
- Convert text into vectors using Bag-of-Words.
- Train a simple classifier: Naive Bayes or Logistic Regression.

Deliverables:

- Code that runs end to end.
- Graph showing category distribution.
- Baseline accuracy. For balanced IMDB data, accuracy is acceptable; for imbalanced data, prefer F1.

Grade weight: 15%.

## Part 2 - Text Preprocessing

Goal: build a reusable preprocessing pipeline and test whether cleaning improves performance.

Tasks:

- Tokenization and normalization: lowercase, remove symbols.
- Remove stop words.
- Use stemming or lemmatization.
- Run the Part 1 classifier again and compare before vs. after preprocessing.

Deliverables:

- Reusable `preprocess()` function.
- Short table: accuracy before preprocessing vs. after preprocessing.
- One-sentence conclusion explaining whether cleaning helped and by how much.

Grade weight: 20%.

## Part 3 - Better Features and Classifier Comparison

Goal: improve text representation, compare classifiers, and evaluate fairly.

Tasks:

- Move from Bag-of-Words to TF-IDF.
- Optionally use n-grams.
- Add features such as POS counts, NER counts, or review length.
- Compare classifiers:
  - Naive Bayes
  - Logistic Regression
  - SVM
  - kNN
  - one additional independently researched ML algorithm
- Evaluate using cross-validation.
- Include confusion matrix, precision, recall, and F1.

Deliverables:

- Classifier comparison table.
- Confusion matrix for the best classifier.
- Short explanation of why the best classifier won.

Grade weight: 25%.

## Part 4 - Improve and Analyze the Best Classifier

Goal: take the best classifier from Part 3, improve it, and understand its errors.

Tasks:

- Try measured improvements:
  - bigrams or other n-grams
  - hyperparameter tuning
  - feature combinations
- Measure every change separately.
- Use cross-validation.
- Inspect which words the classifier relies on most, using high and low feature weights where applicable.
- Collect reviews where the model was wrong, including a few written manually, and identify error patterns.

Deliverables:

- Table: what was tried and how much it changed accuracy/F1.
- List of the most influential words for each category.
- Five analyzed errors.
- Conclusion explaining the limits of the classical approach.

Grade weight: 20%.

## Part 5 - Pretrained Model

Goal: use a pretrained language model and compare it with the classical models.

Tasks:

- Run a pretrained sentiment model using HuggingFace `pipeline`.
- Use the original text, not the processed/stemmed text from Part 2, because the model has its own tokenizer.
- Compare it against the baseline and the best classical classifier.
- Run it specifically on the five errors from Part 4.
- Explain what the pretrained model captures that the classical approach may miss, such as context, word order, or world knowledge.

Deliverables:

- Final comparison table containing all versions from Parts 1-5.
- Note which errors were solved and which were not.
- Summary paragraph: which approach won and at what cost in runtime, complexity, and transparency.

Optional extension:

- Use model embeddings with feature extraction and train a classical classifier on top.
- Or perform full fine-tuning.

Grade weight: 20%.

## Dataset Requirements

Recommended dataset:

- IMDB Movie Reviews
- Task: positive vs. negative sentiment
- Source: Kaggle or HuggingFace
- Used for Parts 1-5

Alternatives:

- SMS Spam
- Tweet Sentiment
- Any other binary text classification dataset from Kaggle

Notes:

- All five parts should use the same dataset so results are directly comparable.
- A ready cleaned dataset may be available from the lecturer.
- IMDB has 50,000 reviews. For practical runtime, use a subset of 3,000-5,000 reviews for the whole project.
- If the dataset is imbalanced, such as SMS Spam with about 13% spam, accuracy alone is misleading; use F1.

## Environment

Suggested installation:

```bash
pip install scikit-learn pandas spacy nltk transformers xgboost
python -m spacy download en_core_web_sm
```

For NLTK resources:

```python
import nltk
nltk.download(["punkt", "stopwords", "wordnet"])
```

Part 5 is easier with GPU. If the rest of the project is in PyCharm, the pretrained-model section can still be run in free Google Colab and the results copied back. Without GPU, use pretrained feature extraction and train a lighter classifier on top.

## Final Submission

Submit:

- One clean documented code project covering all five parts.
- Final comparison table of all versions, including accuracy and F1.
- Short report of 2-3 pages of text:
  - decisions made
  - error analysis
  - what was learned
- Tables and graphs should go in an appendix and do not count toward the 2-3 page limit.

Submission metadata in the HTML was left blank:

- Due date: not specified.
- Submission method: not specified.

## Pre-Submission Checklist

- Code runs end to end without errors in PyCharm or Notebook.
- Cross-validation is used from Part 3 onward.
- One extra ML algorithm not taught in class is included in the comparison table.
- Part 5 uses original text for the pretrained model, not preprocessed text.
- Final comparison table includes all versions.
- Report is 2-3 pages, with tables/graphs in an appendix.
- Both team members understand every part before the defense.

## Grading

| Part | Topic | Weight |
|---|---:|---:|
| 1 | Baseline | 15% |
| 2 | Preprocessing | 20% |
| 3 | Features and classifiers | 25% |
| 4 | Improvement and analysis | 20% |
| 5 | Pretrained model | 20% |

Mandatory cross-validation and the additional ML algorithm are included in Parts 3-4; missing them lowers the grade.

## Useful Reading From The Assignment

- StatQuest: https://www.youtube.com/@statquest
- Introduction to Statistical Learning: https://www.statlearning.com
- scikit-learn User Guide: https://scikit-learn.org/stable/user_guide.html
- spaCy 101: https://spacy.io/usage/spacy-101
- spaCy linguistic features: https://spacy.io/usage/linguistic-features
- NLTK Book: https://www.nltk.org/book/
- sklearn text feature extraction: https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction
- sklearn cross-validation: https://scikit-learn.org/stable/modules/cross_validation.html
- sklearn metrics: https://scikit-learn.org/stable/modules/model_evaluation.html
- HuggingFace LLM/NLP Course: https://huggingface.co/learn/llm-course
- The Illustrated Transformer: https://jalammar.github.io/illustrated-transformer/
- HuggingFace pipeline docs: https://huggingface.co/docs/transformers/main_classes/pipelines
- pandas Getting Started: https://pandas.pydata.org/docs/getting_started/index.html
- Kaggle: https://www.kaggle.com
