# Sentiment Analyzer Failure & Analysis Report

## 1. Overall Accuracy vs. Per-Category Challenge Accuracy
* **Held-Out Test Set Accuracy (In-Domain Movie Reviews):** 80.75%
* **Challenge Set Overall Accuracy:** 64.0% (16/25)
  * **3.1 Negation:** 60% (3/5)
  * **3.2 Sarcasm:** 40% (2/5)
  * **3.3 Mixed Sentiment:** 40% (2/5)
  * **3.4 Intensity:** 80% (4/5)
  * **3.5 Domain Shift:** 100% (5/5)

## 2. Which Linguistic Phenomenon Hurt Most and Why
**Sarcasm (40%)** and **Mixed Sentiment (40%)** hurt the model the hardest, closely followed by **Negation (60%)**. Every single failure traces directly back to the core limitation of a TF-IDF bag-of-words representation: **word order and sentence structure are discarded before the model ever sees the text**.

* **Sarcasm:** Sarcastic reviews use overwhelmingly positive words (*"wonderful"*, *"brilliant"*, *"masterpiece"*) to express a negative meaning. Because the classifier only counts isolated word frequencies without world knowledge or sequential context, those positive keywords push the probability heavily toward `pos` (causing failures on *"Oh wonderful, another two hours..."* and *"Brilliant job making a thriller with zero suspense"*).
* **Mixed Sentiment:** When a review praises one aspect while condemning the whole movie (*"The lead actor gave an Oscar-worthy performance, though the ending ruined everything"*), the bag-of-words representation has no way to know which clause carries the final verdict.
* **Negation:** To a bag-of-words model, *"not good"* and *"good not"* produce the exact same input because word order was stripped away. While it caught negative reviews containing negative words, it failed completely on double-negatives/litotes like *"The movie was not bad, actually quite enjoyable"* and *"I did not hate this film"* because `"not"` and `"bad"`/`"hate"` were treated as independent unigrams rather than a connected phrase.

## 3. How to Fix Negation Specifically
To fix negation specifically without abandoning classical machine learning, I would change the vectorizer from single words (unigrams) to **bigrams (`ngram_range=(1, 2)` in `TfidfVectorizer`)**. Preserving adjacent word pairs turns `"not bad"`, `"not hate"`, and `"not good"` into their own distinct features separate from `"bad"`, `"hate"`, and `"good"`, allowing the model to learn the true polarity of negated phrases.

**What It Would Cost:** Adding bigrams dramatically expands the vocabulary (feature space), which increases memory consumption, slows down training, and introduces data sparsity—many two-word pairs appear only once or twice in 1,600 training reviews, increasing the risk of overfitting unless `max_features` or `min_df` is carefully tuned. (Alternatively, a pre-trained transformer sequence model would read the full sentence in order and solve negation, sarcasm, and mixed sentiment together, at the cost of much higher compute requirements and slower inference latency).

## 4. Deployment Decision: Would I Deploy This?
**No, I would not deploy this model in production.** If a company used this bag-of-words analyzer to automatically route angry customers to a human support agent, **sarcastic customers would be misclassified as happy (`pos`) and completely ignored**. Because frustrated customers frequently use sarcastic praise (*"Oh wonderful, another broken update!"* or *"Brilliant job losing my order"*), the model would treat their complaints as positive feedback and bury the angriest customers instead of escalating them to a human representative.
