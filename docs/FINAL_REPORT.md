# Final Report - NLP Sentiment Classification

Submitters: Saleem Yousef and Noor Shama

## 1. Project Goal And Dataset

The goal of this project is to build a sentiment-classification system that receives an English movie review and predicts whether the review is positive or negative. The project follows the full workflow required in the assignment: a simple baseline, preprocessing, feature engineering and classifier comparison, model improvement and error analysis, and finally comparison with a pretrained language model.

The dataset used is the IMDB movie review dataset. For practical runtime and fair comparison across all stages, the experiments used a fixed balanced subset of 5,000 reviews from `data/imdb_reviews.csv`: 2,500 positive reviews and 2,500 negative reviews. The average review length in this subset was about 230 words, with the shortest review containing 12 words and the longest containing 1,522 words. The same sampled dataset and `random_state=42` were used across the project to keep results reproducible.

## 2. Part 1 - Baseline

The first system was a simple Bag-of-Words baseline. Reviews were converted into word-count vectors using `CountVectorizer`, and a Logistic Regression classifier was trained on a train/test split. This gave a clear starting point for later comparisons.

The baseline performed strongly: accuracy was 0.851 and macro F1 was 0.851 on a 1,000-review holdout set. This shows that even a simple representation can capture many sentiment signals because movie reviews often include direct sentiment words such as "good", "bad", "great", and "worst". The category-distribution graph in the appendix confirms that the subset was balanced, so accuracy is a reasonable metric here. Macro F1 was still reported throughout the project for consistency.

## 3. Part 2 - Preprocessing

The preprocessing stage normalized the text, tokenized it, removed English stop words, and applied stemming. The same Bag-of-Words Logistic Regression setup was then evaluated before and after preprocessing.

| Version | Accuracy | Macro F1 |
|---|---:|---:|
| Raw Bag-of-Words | 0.851 | 0.851 |
| Preprocessed Bag-of-Words | 0.836 | 0.836 |

Preprocessing reduced performance by about 0.015 accuracy points. This is an important result: cleaning text does not always improve classification. In sentiment analysis, some words that look unimportant can still contribute to sentiment or sentence structure. For example, negation and short functional words can change the meaning of a sentence. Stemming can also merge forms in a way that loses nuance. Based on this measurement, the later stages used stronger feature representations instead of assuming that heavier preprocessing was always beneficial.

## 4. Part 3 - Features And Classifier Comparison

Part 3 moved from Bag-of-Words to TF-IDF features and added simple review-length features. The models were evaluated with 5-fold cross-validation, as required from Part 3 onward. The compared classifiers were Naive Bayes, Logistic Regression, SVM, kNN, and Random Forest. Random Forest was included as the additional independently researched ML algorithm beyond the required classifier list.

| Classifier | Mean Accuracy | Mean Macro F1 |
|---|---:|---:|
| Logistic Regression | 0.866 | 0.866 |
| Naive Bayes | 0.858 | 0.858 |
| SVM | 0.857 | 0.857 |
| Random Forest | 0.830 | 0.830 |
| kNN | 0.506 | 0.355 |

Logistic Regression performed best. This is expected for sparse high-dimensional text features because linear classifiers often work very well with TF-IDF. Naive Bayes and SVM were also strong, but slightly lower. Random Forest performed worse, probably because tree-based models are not as naturally suited to very sparse text vectors. kNN performed very poorly because distance-based methods struggle in high-dimensional sparse spaces, where many examples appear similarly far apart.

## 5. Part 4 - Improvement And Error Analysis

Part 4 took the best classical direction and tried measured improvements. The tested variants included TF-IDF unigrams, TF-IDF bigrams, a limited-feature bigram model, and a tuned Logistic Regression model with stronger regularization parameter `C=2.0`.

| Variant | Mean Accuracy | Mean Macro F1 |
|---|---:|---:|
| TF-IDF bigrams + tuned C | 0.867 | 0.867 |
| TF-IDF bigrams | 0.866 | 0.866 |
| TF-IDF unigrams | 0.859 | 0.859 |
| TF-IDF bigrams limited features | 0.850 | 0.850 |

The best classical model was TF-IDF bigrams with tuned Logistic Regression. The improvement over the Part 3 model was small but measurable. Bigrams helped because sentiment sometimes depends on short phrases, not only individual words. For example, phrases like "the worst" are stronger indicators than separate unigram counts.

The influential-word analysis also supported this interpretation. Strong negative features included `bad`, `worst`, `poor`, `the worst`, `awful`, `horrible`, `waste`, `dull`, `boring`, and `terrible`. Strong positive features included `excellent`, `great`, `perfect`, `wonderful`, `beautiful`, `strong`, `amazing`, `love`, `fun`, `enjoyed`, `hilarious`, `best`, and `fantastic`. This makes the model relatively easy to explain because the strongest features match human intuition about sentiment.

The error analysis showed the limits of the classical approach. Several mistakes involved mixed sentiment or contrast, where a review contained both positive and negative language. For example, one negative review described some appealing context but concluded that the movie was predictable and not serious enough; the model predicted positive. Another positive review included complaints about a rushed ending but was overall favorable; the model predicted negative. Other mistakes involved negation or complex phrasing. These examples show that TF-IDF is strong but mostly relies on local word and phrase evidence. It does not truly understand context, sarcasm, or how sentiment changes across a long review.

## 6. Part 5 - Pretrained Model

Part 5 used the HuggingFace `distilbert-base-uncased-finetuned-sst-2-english` sentiment pipeline. This model was evaluated on raw original reviews, not on the preprocessed or stemmed text, because pretrained transformer models have their own tokenizer and were trained on natural text.

| Version | Evaluated Rows | Accuracy | Macro F1 |
|---|---:|---:|---:|
| Part 1 Bag-of-Words baseline | 1000 | 0.851 | 0.851 |
| Part 4 best classical model | 1000 | 0.861 | 0.861 |
| Part 5 HuggingFace pretrained model | 500 | 0.878 | 0.878 |

The pretrained model achieved the best result: 0.878 accuracy and 0.878 macro F1 on the evaluated subset. This improvement is reasonable because a transformer model can use word order and context better than Bag-of-Words or TF-IDF. It can also handle some cases where the meaning depends on the sentence structure rather than isolated words.

The tradeoff is cost. The pretrained model requires additional dependencies, model download time, more memory, and longer runtime. It is also less transparent than Logistic Regression over TF-IDF features. The classical model is easier to explain and still performs strongly, while the pretrained model gives better accuracy at higher complexity.

## 7. Final Conclusion

The best overall model was the pretrained HuggingFace DistilBERT sentiment model, with macro F1 around 0.878. The best classical model was TF-IDF bigrams with tuned Logistic Regression, with macro F1 around 0.867. The difference is not huge, which shows that classical ML is still a strong baseline for sentiment classification.

The project demonstrates the practical tradeoff between interpretability and performance. Classical TF-IDF + Logistic Regression is fast, simple, reproducible, and easy to explain using influential words. The pretrained transformer model performs better and captures more context, but it is heavier and less transparent. For a production system where accuracy is most important, the pretrained model would be preferred. For a lightweight, explainable baseline, the tuned classical model is a strong choice.

## Appendix References

The following generated files should be included as appendix material or screenshots in the final submitted PDF:

- Category distribution graph: `outputs/part1/category_distribution.png`
- Dataset summary: `outputs/part1/dataset_summary.csv`
- Part 1 metrics: `outputs/part1/metrics.json`
- Preprocessing comparison: `outputs/part2/preprocessing_comparison.csv`
- Classifier comparison: `outputs/part3/classifier_comparison.csv`
- Confusion matrix: `outputs/part3/best_confusion_matrix.png`
- Improvement table: `outputs/part4/improvement_table.csv`
- Influential words: `outputs/part4/influential_words.csv`
- Error analysis: `outputs/part4/error_analysis.csv`
- Final comparison: `outputs/part5/final_comparison.csv`
- Pretrained recheck of classical errors: `outputs/part5/part4_error_recheck.csv`