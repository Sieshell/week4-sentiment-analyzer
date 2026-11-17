import random
import nltk
from nltk.corpus import movie_reviews
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Ensure dataset is available
nltk.download('movie_reviews', quiet=True)

def train_and_evaluate():
    # 1. Load data
    documents = [
        (movie_reviews.raw(fileid), category)
        for category in movie_reviews.categories()
        for fileid in movie_reviews.fileids(category)
    ]
    
    # Part 1: Report class counts and balance
    pos_count = sum(1 for _, label in documents if label == 'pos')
    neg_count = sum(1 for _, label in documents if label == 'neg')
    print("=== PART 1: DATASET CLASS BALANCE ===")
    print(f"Positive reviews: {pos_count}")
    print(f"Negative reviews: {neg_count}")
    print("Balance check: The dataset is perfectly balanced (50% positive, 50% negative), so neither class outnumbers the other.\n")

    random.seed(42)
    random.shuffle(documents)

    texts = [doc[0] for doc in documents]
    labels = [doc[1] for doc in documents]

    # Split into 80% training and 20% testing
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    # Part 2: Vectorize (TF-IDF) and train classifier
    vectorizer = TfidfVectorizer(lowercase=True, max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    clf = MultinomialNB()
    clf.fit(X_train_vec, y_train)

    # Evaluate on held-out test set
    y_pred = clf.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=['neg', 'pos'])

    print("=== PART 2: HELD-OUT TEST EVALUATION ===")
    print(f"Test Set Accuracy: {acc:.2%}")
    print("Confusion Matrix (Rows = Actual [neg, pos], Cols = Predicted [neg, pos]):")
    print(cm)
    print("\nWhat the model actually sees:")
    print("With a TF-IDF bag-of-words representation, the model only sees which individual words appear and their weighted frequency—and nothing about the order they came in.\n")

    return vectorizer, clf

def predict_reviews(vectorizer, clf, review_list):
    vec = vectorizer.transform(review_list)
    return clf.predict(vec)

if __name__ == "__main__":
    train_and_evaluate()
