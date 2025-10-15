import streamlit as st
import joblib
import nltk
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import base64

# ----- BACKGROUND IMAGE FUNCTION -----
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url('https://img.freepik.com/premium-vector/cute-wallpaper-background-pastel-colour_493693-246.jpg');
        background-size: cover;
        background-position: center;
    }}
    </style>
    """, unsafe_allow_html=True)

# ----- NLTK SETUP -----
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return " ".join(tokens)

vectorizer = joblib.load("tfidf_vectorizer.pkl")
model = joblib.load("sentiment_model.pkl")
label_mapping = {"Positive": "😂 Positive", "Neutral": "😐 Neutral", "Negative": "👺 Negative"}

st.title("📬 Senticore - Sentiment Analysis")
st.header("Single Review Prediction")
user_input = st.text_area("Enter a review:")

if st.button("Predict Sentiment"):
    if user_input.strip():
        processed = preprocess_text(user_input)
        vect = vectorizer.transform([processed])
        pred = model.predict(vect)[0]
        st.success(f"Prediction: {label_mapping[pred]}")
    else:
        st.warning("Please enter some text.")
