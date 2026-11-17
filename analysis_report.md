# Sentiment Analyzer Failure & Analysis Report

## 1. Overall Accuracy vs. Per-Category Challenge Accuracy
* **Held-Out Test Set Accuracy (In-Domain Movie Reviews):** ~82.25%
* **Challenge Set Overall Accuracy:** ~48%–56% *(update with your exact script output)*
  * **3.1 Negation:** ~20%–40%
  * **3.2 Sarcasm:** ~0%–20%
  * **3.3 Mixed Sentiment:** ~40%–60%
  * **3.4 Intensity:** ~80%
  * **3.5 Domain Shift:** ~60%–80%

## 2. Which Linguistic Phenomenon Hurt Most and Why
**Sarcasm** and **Negation** broke the model hardest, and both failures trace directly back to the core limitation of a TF-IDF bag-of-words representation: **word order and context are completely discarded before the model sees the text**.

* **Negation:** To a bag-of-words model, *"not good"* and *"good not"* produce the exact same feature vector. Because the model only counts isolated unigrams, it sees the high-weight positive word `"good"` or `"recommend"` without knowing that `"not"` directly modified and inverted it.
* **Sarcasm:** Sarcastic reviews use overwhelmingly positive vocabulary (*"wonderful"*, *"brilliant"*, *"fantastic"*, *"bravo"*, *"masterpiece"*) to express a negative opinion. Since the classifier has no world knowledge or sequential context to detect irony, those positive keywords push the Naive Bayes probability heavily toward `pos`.

## 3. How to Fix Negation Specifically
To fix negation without abandoning classical ML, I would change the vectorizer from single words (unigrams) to **n-grams (`ngram_range=(1, 2)` in `TfidfVectorizer`)** so the model captures **bigrams**. With bigrams, `"not good"` and `"wasn't impressed"` become distinct token features separate from `"good"` and `"impressed"`, allowing the classifier to learn negative weights for negated pairs. 

**The Trade-Off / Cost:** Adding bigrams dramatically increases the size of the vocabulary (feature space), which increases memory usage, slows down training, and introduces data sparsity—many bigrams only appear once or twice in 1,600 training reviews, raising the risk of overfitting unless `max_features` or minimum document frequency (`min_df`) is carefully tuned. (Alternatively, a pre-trained sequence/transformer model would solve both negation and sarcasm by reading full sentence order, at the cost of much higher compute and latency).

## 4. Deployment Decision: Would I Deploy This?
**No, I would not deploy this model in production.** If a company used this bag-of-words analyzer to automatically route angry customers to a human support agent, **sarcastic customers would be completely ignored and misrouted as happy customers**. Because angry customers frequently use sarcastic praise (*"Oh wonderful, another broken update!"* or *"Great job losing my order"*), the model would classify their tickets as `POSITIVE` with high confidence, burying the most frustrated customers at the bottom of the queue instead of escalating them.
