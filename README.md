📬 Senticore - Sentiment Analysis Web App

Senticore is an interactive Streamlit-based web application that analyzes the sentiment of user-entered text (such as reviews, feedback, or comments). It uses Natural Language Processing (NLP) and Machine Learning (ML) to determine whether the sentiment of the input text is Positive, Negative, or Neutral.

💡 Overview

This project allows users to input any review or sentence, and it will instantly predict the emotional tone behind the text. The model processes the text using TF-IDF vectorization and predicts the sentiment using a pre-trained machine learning model. The app features a beautiful pastel-themed UI and uses emojis for an engaging user experience.

🧠 Features

✅ Real-Time Sentiment Prediction – Enter any review and get an instant result.
✅ Clean Text Preprocessing – Automatically cleans and lemmatizes text before prediction.
✅ User-Friendly Interface – Built using Streamlit with a soft pastel background.
✅ Pre-Trained Model – Uses a saved ML model (sentiment_model.pkl) and TF-IDF vectorizer (tfidf_vectorizer.pkl).
✅ Emoji-Based Output – Sentiment predictions are displayed with fun emojis:

😂 Positive

😐 Neutral

👺 Negative

⚙️ Technologies Used

Python 3.x

Streamlit – for building the web interface

NLTK – for text preprocessing (tokenization, stopword removal, lemmatization)

Scikit-learn – for machine learning and TF-IDF vectorization

Joblib – for loading the trained model and vectorizer

Pandas, Regex – for text handling and cleaning

🧩 How It Works

The user enters a review or any text into the input box.

The text is preprocessed:

Converted to lowercase

URLs and special characters removed

Tokenized and lemmatized

Stopwords removed

The processed text is converted into TF-IDF features using the saved vectorizer.

The pre-trained ML model predicts the sentiment.

The result is displayed on the screen with an appropriate emoji.

🚀 How to Run Locally

Clone this repository:

git clone https://github.com/yourusername/Senticore.git
cd Senticore
Install dependencies:

pip install -r requirements.txt
Run the Streamlit app:
streamlit run app.py
Open your browser and go to:
👉 http://localhost:8501

📂 Project Structure
Senticore/
│
├── app.py                    # Main Streamlit application file
├── sentiment_model.pkl        # Pre-trained sentiment analysis model
├── tfidf_vectorizer.pkl       # TF-IDF vectorizer used for text transformation
├── requirements.txt           # List of dependencies
└── README.md                  # Project documentation

🎯 Use Case

This app can be used by:

Businesses analyzing customer feedback or product reviews

Students and researchers learning NLP and sentiment analysis

Developers building AI-driven feedback systems

Anyone interested in understanding emotional tones in text data

💬 Example
Input Text	Prediction
"I love this product, it’s amazing!"	😂 Positive
"It’s okay, not too bad."	😐 Neutral
"Worst experience ever, totally disappointed!"	👺 Negative
🌈 Future Improvements

Add batch sentiment analysis for multiple reviews.

Integrate data visualization (e.g., sentiment distribution charts).

Allow users to upload CSV files for bulk review analysis.

Support multilingual sentiment detection.
