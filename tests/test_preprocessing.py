import pytest
import pandas as pd
from sklearn.model_selection import train_test_split
from src.preprocessing import load_data, clean_text, preprocess_data, save_vectorizer, load_vectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import os
import pickle


@pytest.fixture
def sample_data():
    """Fixture to provide a small dataset for testing."""
    data = {
        'review': ['I love this movie!', 'I hate this movie!', 'It was an amazing movie.', 'The movie was terrible.'],
        'sentiment': ['positive', 'negative', 'positive', 'negative']
    }
    df = pd.DataFrame(data)
    return df


def test_load_data(tmp_path):
    """Test loading data from a CSV file."""
    # Create a sample CSV file
    file_path = tmp_path / "sample.csv"
    data = {'review': ['Sample review 1', 'Sample review 2'], 'sentiment': ['positive', 'negative']}
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)

    # Load the data
    loaded_df = load_data(file_path)
    assert not loaded_df.empty
    assert list(loaded_df.columns) == ['review', 'sentiment']


def test_clean_text():
    """Test text cleaning function."""
    text = "I love this movie! <br /> It was amazing!!!"
    cleaned_text = clean_text(text)
    assert cleaned_text == "i love this movie  it was amazing"


def test_preprocess_data(sample_data):
    """Test the preprocessing function with both CountVectorizer and TfidfVectorizer."""
    # Using TF-IDF Vectorizer
    (X_train, X_test, y_train, y_test), vectorizer = preprocess_data(sample_data, vectorizer='tfidf')
    assert X_train.shape[0] == 3  # 80% of 4 samples
    assert X_test.shape[0] == 1  # 20% of 4 samples
    assert isinstance(vectorizer, TfidfVectorizer)

    # Using Count Vectorizer
    (X_train, X_test, y_train, y_test), vectorizer = preprocess_data(sample_data, vectorizer='count')
    assert isinstance(vectorizer, CountVectorizer)


def test_save_and_load_vectorizer(tmp_path):
    """Test saving and loading vectorizer."""
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    file_path = tmp_path / "vectorizer.pkl"

    # Save the vectorizer
    save_vectorizer(vectorizer, file_path)

    # Load the vectorizer
    loaded_vectorizer = load_vectorizer(file_path)
    assert isinstance(loaded_vectorizer, TfidfVectorizer)


def test_train_test_split(sample_data):
    """Test the train/test split size."""
    (X_train, X_test, y_train, y_test), _ = preprocess_data(sample_data, vectorizer='tfidf')
    assert len(X_train) == 3  # 80% of the data
    assert len(X_test) == 1  # 20% of the data
    assert len(y_train) == 3
    assert len(y_test) == 1
