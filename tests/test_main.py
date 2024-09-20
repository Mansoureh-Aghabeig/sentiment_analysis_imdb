import pytest
from unittest.mock import patch, MagicMock
import io
import sys
from src.main import main

@patch('src.main.load_data')
@patch('src.main.preprocess_data')
@patch('src.main.save_vectorizer')
@patch('src.main.train_logistic_regression')
@patch('src.main.train_naive_bayes')
@patch('src.main.train_random_forest')
@patch('src.main.save_model')
@patch('src.main.evaluate_model')
@patch('src.main.plot_confusion_matrix')
def test_main_logistic_regression(mock_plot_confusion_matrix, mock_evaluate_model, mock_save_model, mock_train_random_forest, mock_train_naive_bayes, mock_train_logistic_regression, mock_save_vectorizer, mock_preprocess_data, mock_load_data):
    # Mock return values
    mock_load_data.return_value = MagicMock()
    mock_preprocess_data.return_value = (MagicMock(), MagicMock())
    mock_train_logistic_regression.return_value = MagicMock()
    mock_train_naive_bayes.return_value = MagicMock()
    mock_train_random_forest.return_value = MagicMock()
    mock_save_vectorizer.return_value = None
    mock_save_model.return_value = None
    mock_evaluate_model.return_value = (0.9, 0.8, 0.7)  # example metrics

    # Capture the output of the print statements
    with patch('sys.stdout', new=io.StringIO()) as fake_out:
        with patch('builtins.input', side_effect=['1']):
            main()
            output = fake_out.getvalue()

    # Check if the expected output is in the captured output
    assert "Training and evaluating Logistic Regression..." in output
    assert "Accuracy: 0.9000" in output
    assert "Precision: 0.8000" in output
    assert "Recall: 0.7000" in output

@patch('src.main.load_data')
@patch('src.main.preprocess_data')
@patch('src.main.save_vectorizer')
@patch('src.main.train_logistic_regression')
@patch('src.main.train_naive_bayes')
@patch('src.main.train_random_forest')
@patch('src.main.save_model')
@patch('src.main.evaluate_model')
@patch('src.main.plot_confusion_matrix')
def test_main_naive_bayes(mock_plot_confusion_matrix, mock_evaluate_model, mock_save_model, mock_train_random_forest, mock_train_naive_bayes, mock_train_logistic_regression, mock_save_vectorizer, mock_preprocess_data, mock_load_data):
    # Mock return values
    mock_load_data.return_value = MagicMock()
    mock_preprocess_data.return_value = (MagicMock(), MagicMock())
    mock_train_naive_bayes.return_value = MagicMock()
    mock_train_logistic_regression.return_value = MagicMock()
    mock_train_random_forest.return_value = MagicMock()
    mock_save_vectorizer.return_value = None
    mock_save_model.return_value = None
    mock_evaluate_model.return_value = (0.9, 0.8, 0.7)  # example metrics

    # Capture the output of the print statements
    with patch('sys.stdout', new=io.StringIO()) as fake_out:
        with patch('builtins.input', side_effect=['2']):
            main()
            output = fake_out.getvalue()

    # Check if the expected output is in the captured output
    assert "Training and evaluating Naive Bayes..." in output
    assert "Accuracy: 0.9000" in output
    assert "Precision: 0.8000" in output
    assert "Recall: 0.7000" in output

@patch('src.main.load_data')
@patch('src.main.preprocess_data')
@patch('src.main.save_vectorizer')
@patch('src.main.train_logistic_regression')
@patch('src.main.train_naive_bayes')
@patch('src.main.train_random_forest')
@patch('src.main.save_model')
@patch('src.main.evaluate_model')
@patch('src.main.plot_confusion_matrix')
def test_main_random_forest(mock_plot_confusion_matrix, mock_evaluate_model, mock_save_model, mock_train_random_forest, mock_train_naive_bayes, mock_train_logistic_regression, mock_save_vectorizer, mock_preprocess_data, mock_load_data):
    # Mock return values
    mock_load_data.return_value = MagicMock()
    mock_preprocess_data.return_value = (MagicMock(), MagicMock())
    mock_train_random_forest.return_value = MagicMock()
    mock_train_naive_bayes.return_value = MagicMock()
    mock_train_logistic_regression.return_value = MagicMock()
    mock_save_vectorizer.return_value = None
    mock_save_model.return_value = None
    mock_evaluate_model.return_value = (0.9, 0.8, 0.7)  # example metrics

    # Capture the output of the print statements
    with patch('sys.stdout', new=io.StringIO()) as fake_out:
        with patch('builtins.input', side_effect=['3']):
            main()
            output = fake_out.getvalue()

    # Check if the expected output is in the captured output
    assert "Training and evaluating Random Forest..." in output
    assert "Accuracy: 0.9000" in output
    assert "Precision: 0.8000" in output
    assert "Recall: 0.7000" in output

