import pandas as pd
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline # Useful for chaining steps
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os # To handle potential issues with data download/path

def load_and_prepare_data(selected_categories=None, subset='train', shuffle=True, random_state=42):
    """
    Fetches the 20 Newsgroups dataset, optionally filtered by categories.

    Args:
        selected_categories (list, optional): A list of category names to include.
                                              If None, all categories are loaded.
        subset (str): Which subset of the dataset to load: 'train', 'test', or 'all'.
        shuffle (bool): Whether to shuffle the data.
        random_state (int): Seed for reproducibility.

    Returns:
        sklearn.utils.Bunch: A scikit-learn Bunch object containing data, target,
                             and target_names.
    """
    print(f"Loading 20 Newsgroups dataset (subset: '{subset}', categories: {selected_categories})...")
    try:
        newsgroups_data = fetch_20newsgroups(
            subset=subset,
            categories=selected_categories,
            shuffle=shuffle,
            random_state=random_state
        )
        print(f"Loaded {len(newsgroups_data.data)} documents for {len(newsgroups_data.target_names)} categories.")
        return newsgroups_data
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please check your internet connection or try again later.")
        return None

def train_text_classifier(training_data):
    """
    Trains a text classification model using a Pipeline for vectorization and classification.
    The pipeline includes CountVectorizer, TfidfTransformer, and MultinomialNB.

    Args:
        training_data (sklearn.utils.Bunch): The training dataset with 'data' and 'target'.

    Returns:
        sklearn.pipeline.Pipeline: The trained scikit-learn Pipeline model.
    """
    print("\n--- Training the Text Classifier ---")
    # A Pipeline chains multiple estimators into one.
    # This simplifies the workflow and ensures consistent preprocessing.
    text_clf = Pipeline([
        ('vect', CountVectorizer()),        # Step 1: Convert text documents to a matrix of token counts
        ('tfidf', TfidfTransformer()),      # Step 2: Transform counts to normalized TF-IDF features
        ('clf', MultinomialNB()),           # Step 3: Train a Multinomial Naive Bayes classifier
    ])

    # Fit the pipeline to the training data
    # The .fit() method of the pipeline handles calling fit_transform on early steps
    # and fit on the final estimator.
    fitted_model = text_clf.fit(training_data.data, training_data.target)
    print("Model training complete.")
    return fitted_model

def evaluate_model(model, test_data):
    """
    Evaluates the trained text classification model on test data.

    Args:
        model (sklearn.pipeline.Pipeline): The trained scikit-learn Pipeline model.
        test_data (sklearn.utils.Bunch): The test dataset with 'data' and 'target'.
    """
    print("\n--- Evaluating Model Performance ---")
    # Make predictions on the test data
    predicted_target = model.predict(test_data.data)

    # Print evaluation metrics
    print("Accuracy Score:", accuracy_score(test_data.target, predicted_target))
    print("\nConfusion Matrix:")
    print(confusion_matrix(test_data.target, predicted_target))
    print("\nClassification Report:")
    print(classification_report(test_data.target, predicted_target, target_names=test_data.target_names))

def predict_new_documents(model, new_docs, target_names):
    """
    Predicts the category for new, unseen documents.

    Args:
        model (sklearn.pipeline.Pipeline): The trained scikit-learn Pipeline model.
        new_docs (list): A list of strings, where each string is a document to classify.
        target_names (list): A list of category names corresponding to the target labels.
    """
    print("\n--- Predicting Categories for New Documents ---")
    predicted_categories_indices = model.predict(new_docs)

    for i, doc in enumerate(new_docs):
        predicted_category_name = target_names[predicted_categories_indices[i]]
        print(f"Document: '{doc[:60]}...' --------> Predicted Category: {predicted_category_name}")

if __name__ == "__main__":
    # Define the categories of interest
    selected_categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']

    # 1. Load Training Data
    training_data = load_and_prepare_data(selected_categories=selected_categories, subset='train')
    if training_data is None:
        exit() # Exit if data loading failed

    # Load Test Data (for proper evaluation later)
    test_data = load_and_prepare_data(selected_categories=selected_categories, subset='test')
    if test_data is None:
        exit() # Exit if data loading failed

    # 2. Train the Model
    # The pipeline handles text vectorization (CountVectorizer, TfidfTransformer)
    # and then applies the classifier (MultinomialNB).
    text_classifier_model = train_text_classifier(training_data)

    # 3. Evaluate the Model on Test Data
    evaluate_model(text_classifier_model, test_data)

    # 4. Make Predictions on New Unseen Documents
    new_documents_to_predict = [
        'My favourite topic has something to do with quantum physics and quantum mechanics',
        'This has nothing to do with church or religion',
        'Software engineering is getting hotter and hotter nowadays',
        'Is there a cure for the common cold or flu?',
        'Does God exist in the universe?',
        'How to optimize rendering performance in OpenGL?'
    ]
    predict_new_documents(text_classifier_model, new_documents_to_predict, training_data.target_names)
