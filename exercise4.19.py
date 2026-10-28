
# Exercise 4.19 - Sentiment Analysis
# Modified from Example 4.23

from transformers import pipeline

# Load sentiment analysis model
classifier = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)

# Different sentences: positive and negative
sentences = [
    "This movie is amazing and I really enjoyed it.",
    "The food was delicious and the service was excellent.",
    "I am very happy with the results of this project.",
    "This movie is terrible and very boring.",
    "The food was awful and the service was disappointing.",
    "I am frustrated because the application keeps crashing."
]

# Sentiment analysis
results = classifier(sentences)

print("\n===== SENTIMENT ANALYSIS RESULTS =====")

for i, (sentence, result) in enumerate(
    zip(sentences, results), start=1
):
    print(f"\nSentence {i}: {sentence}")
    print(f"Sentiment : {result['label']}")
    print(f"Confidence: {result['score'] * 100:.2f}%")

# Count positive and negative sentiments
positive = sum(r['label'] == 'POSITIVE' for r in results)
negative = sum(r['label'] == 'NEGATIVE' for r in results)

print("\n===== SENTIMENT SUMMARY =====")
print(f"Positive Sentences: {positive}")
print(f"Negative Sentences: {negative}")
print(f"Total Sentences   : {len(sentences)}")
