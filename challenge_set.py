from sentiment_analyzer import train_and_evaluate, predict_reviews

# 5 custom reviews per category (25 total) labeled 'pos' or 'neg'
challenge_data = {
    "3.1 Negation": [
        ("The acting was not good at all.", "neg"),
        ("I wasn't impressed by the story or the pacing.", "neg"),
        ("This isn't a film I'd recommend to anyone.", "neg"),
        ("The movie was not bad, actually quite enjoyable.", "pos"),
        ("I did not hate this film; it surprised me.", "pos"),
    ],
    "3.2 Sarcasm": [
        ("Oh wonderful, another two hours of my life I will never get back.", "neg"),
        ("Brilliant job making a thriller with zero suspense.", "neg"),
        ("Yeah, watching paint dry is definitely more exciting than this masterpiece.", "neg"),
        ("What a fantastic way to cure insomnia—fell asleep in ten minutes.", "neg"),
        ("Bravo to the writers for inventing a plot that makes zero sense.", "neg"),
    ],
    "3.3 Mixed Sentiment": [
        ("Beautiful cinematography, completely wasted on a terrible script.", "neg"),
        ("The lead actor gave an Oscar-worthy performance, though the ending ruined everything.", "neg"),
        ("Terrible special effects and cheesy dialogue, yet I genuinely loved every minute.", "pos"),
        ("The pacing drags badly in the middle, but the brilliant finale makes it worth watching.", "pos"),
        ("Great soundtrack and gorgeous costumes, sadly trapped in a boring movie.", "neg"),
    ],
    "3.4 Intensity": [
        ("It was fine.", "pos"),
        ("The film was decent enough for a rainy afternoon.", "pos"),
        ("It was a pretty good movie overall.", "pos"),
        ("This was an extraordinary, breathtaking masterpiece.", "pos"),
        ("It was slightly disappointing.", "neg"),
    ],
    "3.5 Domain Shift (Restaurant & Product Reviews)": [
        ("The pasta was cold and the waiter forgot our drinks twice.", "neg"),
        ("Best wireless headphones I have ever owned, incredible battery life.", "pos"),
        ("The laptop screen flickered out of the box and customer support hung up on me.", "neg"),
        ("Fresh sushi, fast service, and very reasonable prices.", "pos"),
        ("This blender leaked smoothie all over my kitchen counter on day one.", "neg"),
    ],
}

def run_challenge():
    vectorizer, clf = train_and_evaluate()

    print("=== PART 3: CHALLENGE SET ACCURACY BY CATEGORY ===")
    total_correct = 0
    total_count = 0

    for category, items in challenge_data.items():
        texts = [text for text, _ in items]
        true_labels = [label for _, label in items]
        preds = predict_reviews(vectorizer, clf, texts)

        correct = sum(p == t for p, t in zip(preds, true_labels))
        total_correct += correct
        total_count += len(items)
        acc = (correct / len(items)) * 100

        print(f"\n{category}: {correct}/{len(items)} ({acc:.0f}% accuracy)")
        for text, true_l, pred_l in zip(texts, true_labels, preds):
            status = "PASS" if true_l == pred_l else "FAIL"
            print(f"  [{status}] True: {true_l} | Pred: {pred_l} | \"{text}\"")

    print(f"\nOverall Challenge Set Accuracy: {total_correct}/{total_count} ({(total_correct/total_count)*100:.1f}%)")

if __name__ == "__main__":
    run_challenge()
