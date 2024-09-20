from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pickle

def train_naive_bayes(X_train, y_train):
    """Trains a Naive Bayes classifier."""
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    return nb_model

def train_logistic_regression(X_train, y_train):
    """Trains a Logistic Regression classifier."""
    lr_model = LogisticRegression(max_iter=500, solver='lbfgs')  # Increased max_iter to 500
    lr_model.fit(X_train, y_train)
    return lr_model

def train_random_forest(X_train, y_train):
    """Trains a Random Forest classifier."""
    rf_model = RandomForestClassifier(n_estimators=100)
    rf_model.fit(X_train, y_train)
    return rf_model

def save_model(model, file_path):
    """Saves the model to disk."""
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)

def load_model(file_path):
    """Loads a trained model from disk."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)
