import pytest
import numpy as np
from unittest.mock import MagicMock
from src.evaluation import evaluate_model, plot_confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score


@pytest.fixture
def sample_data():
    """Fixture to provide a small test dataset."""
    X_test = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0], [0, 0, 1]])
    y_test = np.array([1, 0, 1, 0])
    return X_test, y_test


@pytest.fixture
def mock_model():
    """Fixture to provide a mock model."""
    model = MagicMock()
    # Mock the predict method of the model
    model.predict.return_value = np.array([1, 0, 1, 0])  # Simulated predictions
    return model


def test_evaluate_model(mock_model, sample_data):
    """Test the evaluate_model function."""
    X_test, y_test = sample_data

    # Expected values
    expected_accuracy = accuracy_score(y_test, mock_model.predict(X_test))
    expected_precision = precision_score(y_test, mock_model.predict(X_test))
    expected_recall = recall_score(y_test, mock_model.predict(X_test))

    # Call the function to test
    accuracy, precision, recall = evaluate_model(mock_model, X_test, y_test)

    # Check if the outputs match the expected values
    assert accuracy == pytest.approx(expected_accuracy), f"Expected accuracy: {expected_accuracy}, but got {accuracy}"
    assert precision == pytest.approx(
        expected_precision), f"Expected precision: {expected_precision}, but got {precision}"
    assert recall == pytest.approx(expected_recall), f"Expected recall: {expected_recall}, but got {recall}"


def test_plot_confusion_matrix(mock_model, sample_data):
    """Test the plot_confusion_matrix function."""
    X_test, y_test = sample_data

    # Ensure that plot_confusion_matrix runs without errors
    try:
        plot_confusion_matrix(mock_model, X_test, y_test)
    except Exception as e:
        pytest.fail(f"plot_confusion_matrix raised an exception: {e}")
