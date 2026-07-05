# Final Report Template

Submit a 2-3 page report. Tables and graphs can go in an appendix.

## 1. Project Goal And Dataset

Describe the goal: classify English IMDB movie reviews as positive or negative.

Mention:

- dataset: IMDB movie reviews
- subset: 5,000 balanced reviews
- labels: positive / negative
- why the same subset was used across all parts

## 2. Baseline

Explain Part 1:

- Bag-of-Words representation
- Logistic Regression baseline
- accuracy and F1 around 0.851
- category distribution graph in appendix

## 3. Preprocessing

Explain Part 2:

- lowercase/tokenization
- stop-word removal
- stemming
- comparison before and after preprocessing
- result: preprocessing reduced accuracy from about 0.851 to 0.836
- possible reason: useful sentiment words or phrase context may have been removed

## 4. Classifier Comparison

Explain Part 3:

- TF-IDF features
- review-length features
- cross-validation
- compared Naive Bayes, Logistic Regression, SVM, kNN, and Random Forest
- Random Forest is the extra independently researched algorithm
- Logistic Regression performed best
- kNN performed badly because sparse high-dimensional text vectors are difficult for distance-based methods

## 5. Improvement And Error Analysis

Explain Part 4:

- tested unigrams, bigrams, limited features, and tuned regularization
- best classical version: TF-IDF bigrams with Logistic Regression `C=2.0`
- discuss influential words from `outputs/part4/influential_words.csv`
- discuss five errors from `outputs/part4/error_analysis.csv`
- conclude where the classical approach struggles: sarcasm, negation, mixed sentiment, context

## 6. Pretrained Model

Explain Part 5:

- used HuggingFace `distilbert-base-uncased-finetuned-sst-2-english`
- model received raw original text, not preprocessed text
- pretrained result: about 0.878 accuracy and 0.878 macro F1 on 500 evaluated reviews
- compare it to the classical model

## 7. Final Conclusion

Suggested conclusion:

The pretrained model achieved the best score, but it is slower, heavier, and less transparent. The classical TF-IDF + Logistic Regression model is still strong, fast, easy to explain, and useful as a baseline. The project shows the tradeoff between interpretability and modern contextual performance.

## Appendix Items

Include:

- category distribution graph
- preprocessing comparison table
- classifier comparison table
- confusion matrix
- improvement table
- influential words table
- five-error analysis table
- final comparison table