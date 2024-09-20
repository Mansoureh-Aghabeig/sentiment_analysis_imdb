import os
from src.preprocessing import load_data, preprocess_data, save_vectorizer
from src.model import train_naive_bayes, train_logistic_regression, train_random_forest, save_model
from src.evaluation import evaluate_model, plot_confusion_matrix

# Change working directory to the root of the project
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_PATH = "data/imdb_reviews.csv"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"
MODEL_DIR = "models/"


def main():
    # Load data
    df = load_data(DATA_PATH)

    # Preprocess data
    (X_train, X_test, y_train, y_test), vectorizer = preprocess_data(df)

    # Save the vectorizer
    save_vectorizer(vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))

    # Ask user to choose a model
    print("Choose a model to train:")
    print("1. Logistic Regression")
    print("2. Naive Bayes")
    print("3. Random Forest")

    model_choice = input("Enter the number corresponding to the model: ")

    if model_choice == '1':
        model_name = 'Logistic Regression'
        model = train_logistic_regression(X_train, y_train)
        print(f"Training and evaluating {model_name}...")
    elif model_choice == '2':
        model_name = 'Naive Bayes'
        model = train_naive_bayes(X_train, y_train)
        print(f"Training and evaluating {model_name}...")
    elif model_choice == '3':
        model_name = 'Random Forest'
        model = train_random_forest(X_train, y_train)
        print(f"Training and evaluating {model_name}...")
    else:
        print("Invalid choice. Please select a valid option.")
        return  # Exit if invalid choice



    # Save the model
    model_path = os.path.join(MODEL_DIR, f"{model_name.lower().replace(' ', '_')}_model.pkl")
    save_model(model, model_path)

    # Evaluate the model
    accuracy, precision, recall = evaluate_model(model, X_test, y_test)

    # Print the evaluation metrics
    print(f"{model_name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print("-" * 30)

    # Plot confusion matrix (can visualize later in Jupyter Lab)
    plot_confusion_matrix(model, X_test, y_test)


if __name__ == "__main__":
    main()
