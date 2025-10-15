# Quick Fix: Create Model Files Script
# Save this as 'create_models.py' and run it first

# To install joblib, run the following command in your terminal:
# pip install joblib

import joblib
import pandas as pd
import numpy as np
# To install scikit-learn, run the following command in your terminal:
# pip install scikit-learn

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import re

def simple_preprocess(text):
    """Simple preprocessing function"""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    return " ".join(tokens)

# Sample data for training
sample_reviews = [
    "This movie is absolutely fantastic! I loved every minute of it.",
    "Terrible movie, waste of time. Would not recommend.",
    "The movie was okay, nothing special but not bad either.",
    "Amazing acting and great storyline. Highly recommended!",
    "Boring and predictable. Very disappointing.",
    "Good movie with excellent cinematography.",
    "Not the best movie I've seen, but it was entertaining.",
    "Outstanding performance by all actors. Must watch!",
    "Average movie, could have been better.",
    "Worst movie ever made. Terrible plot and acting.",
    "Excellent direction and amazing visuals.",
    "The movie is decent, worth watching once.",
    "Fantastic storyline and great character development.",
    "Poor execution and weak script.",
    "Good entertainment value, enjoyed watching it.",
    "Brilliant movie with outstanding performances.",
    "Not impressed with this movie at all.",
    "Great movie, would definitely watch again.",
    "Mediocre film with some good moments.",
    "Exceptional movie, one of the best I've seen.",
    "Love this product! Works perfectly.",
    "Hate it, complete waste of money.",
    "It's alright, does what it says.",
    "Perfect quality, exactly as described.",
    "Disappointed with the purchase.",
] * 4  # Repeat to get more training data

sample_labels = [
    "Positive", "Negative", "Neutral", "Positive", "Negative",
    "Positive", "Neutral", "Positive", "Neutral", "Negative",
    "Positive", "Neutral", "Positive", "Negative", "Positive",
    "Positive", "Negative", "Positive", "Neutral", "Positive",
    "Positive", "Negative", "Neutral", "Positive", "Negative"
] * 4

# Preprocess the data
processed_reviews = [simple_preprocess(review) for review in sample_reviews]

# Create TF-IDF vectorizer
print("Creating TF-IDF vectorizer...")
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X = vectorizer.fit_transform(processed_reviews)

# Create labels
y = np.array(sample_labels)

# Train model
print("Training sentiment model...")
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X, y)

# Save both files
print("Saving model files...")
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
joblib.dump(model, 'sentiment_model.pkl')

print("✅ SUCCESS! Model files created:")
print("   - tfidf_vectorizer.pkl")
print("   - sentiment_model.pkl")
print("\n🚀 Now run: streamlit run app.py")
