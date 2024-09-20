import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import re


def load_data(file_path):
    """Loads IMDB dataset from CSV file."""
    return pd.read_csv(file_path)


def clean_text(text):
    """Performs basic text cleaning: lowercasing, removing special characters."""
    text = text.lower()
    text = re.sub(r'<br />', ' ', text)  # Remove HTML line breaks
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation and numbers
    return text


def preprocess_data(df, column='review', target='sentiment', vectorizer='tfidf'):
    """Cleans the text data and transforms it into a numerical format."""
    df['cleaned_review'] = df[column].apply(clean_text)

    if vectorizer == 'count':
        vectorizer = CountVectorizer(stop_words='english', max_features=5000)
    else:
        vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)

    X = vectorizer.fit_transform(df['cleaned_review'])
    y = df[target].apply(lambda x: 1 if x == 'positive' else 0)

    return train_test_split(X, y, test_size=0.2, random_state=42), vectorizer


def save_vectorizer(vectorizer, file_path):
    """Saves the vectorizer for future use."""
    import pickle
    with open(file_path, 'wb') as f:
        pickle.dump(vectorizer, f)


def load_vectorizer(file_path):
    """Loads the vectorizer."""
    import pickle
    with open(file_path, 'rb') as f:
        return pickle.load(f)
