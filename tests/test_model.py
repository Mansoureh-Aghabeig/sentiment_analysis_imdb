import pytest
import os
import pickle
from sklearn.datasets import make_classification
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from src.model import train_naive_bayes, train_logistic_regression, train_random_forest, save_model, load_model


@pytest.fixture
def sample_data():
    """Fixture to provide a small dataset for testing."""
    X, y = make_classification(n_samples=100, n_features=20, n_classes=2, random_state=42)
    return X, y


def test_train_naive_bayes(sample_data):
    """Test training Naive Bayes model."""
    X_train, y_train = sample_data
    model = train_naive_bayes(X_train, y_train)

    assert isinstance(model, MultinomialNB)
    assert model.classes_.shape[0] == 2  # Ensure binary classification
    assert model.coef_.shape[1] == X_train.shape[1]


def test_train_logistic_regression(sample_data):
    """Test training Logistic Regression model."""
    X_train, y_train = sample_data
    model = train_logistic_regression(X_train, y_train)

    assert isinstance(model, LogisticRegression)
    assert model.classes_.shape[0] == 2
    assert model.coef_.shape[1] == X_train.shape[1]


def test_train_random_forest(sample_data):
    """Test training Random Forest model."""
    X_train, y_train = sample_data
    model = train_random_forest(X_train, y_train)

    assert isinstance(model, RandomForestClassifier)
    assert model.n_estimators == 100
    assert model.classes_.shape[0] == 2


def test_save_and_load_model(tmp_path):
    """Test saving and loading a model."""
    X_train, y_train = make_classification(n_samples=100, n_features=20, n_classes=2, random_state=42)
    model = train_naive_bayes(X_train, y_train)

    # Save the model
    file_path = tmp_path / "naive_bayes_model.pkl"
    save_model(model, file_path)

    # Check that the file was created
    assert os.path.exists(file_path)

    # Load the model back
    loaded_model = load_model(file_path)

    # Ensure the loaded model is the same as the original model
    assert isinstance(loaded_model, MultinomialNB)
    assert loaded_model.classes_.shape[0] == 2
    assert loaded_model.coef_.shape[1] == X_train.shape[1]
