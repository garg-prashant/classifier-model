#!/usr/bin/env python3
"""
Machine Learning Pipeline Module

This module provides a complete machine learning pipeline for binary classification
using synthetic data. It includes functionality for data generation, CSV I/O,
model training with scikit-learn, model persistence, and evaluation.

The pipeline follows these steps:
1. Generate synthetic dataset
2. Save/load data to/from CSV
3. Split data into training and evaluation sets
4. Train a Random Forest classifier
5. Save and load the trained model
6. Make predictions and evaluate performance

"""

import random
import numpy as np
import csv
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def generate_dataset(num_samples=1000):
    """
    Generate a synthetic dataset for binary classification.
    
    Creates a dataset with two features (feature_1, feature_2) and a binary label.
    The classification rule is: label = 1 if feature_1 + feature_2 > 0, else 0.
    
    Args:
        num_samples (int): Number of samples to generate. Defaults to 1000.
        
    Returns:
        list: List of dictionaries containing the dataset, where each dictionary
              has keys 'feature_1', 'feature_2', and 'label'.
    """
    dataset = []
    for i in range(num_samples):
        feature_1 = random.uniform(-10, 10)
        feature_2 = random.uniform(-10, 10)
        
        # Simple classification rule: if feature_1 + feature_2 > 0, label is 1, else 0
        label = 1 if feature_1 + feature_2 > 0 else 0
        
        dataset.append({
            'feature_1': feature_1,
            'feature_2': feature_2,
            'label': label
        })
    
    return dataset


def save_to_csv(data, filename='dataset.csv'):
    """
    Save dataset to a CSV file.
    
    Args:
        data (list): List of dictionaries containing the data to save.
        filename (str): Name of the CSV file. Defaults to 'dataset.csv'.
        
    Raises:
        ValueError: If data is empty.
    """
    if not data:
        raise ValueError("Data cannot be empty")
    
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = data[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in data:
            writer.writerow(row)
    
    print(f"Dataset saved to {filename}")


def read_from_csv(filename='dataset.csv'):
    """
    Read dataset from a CSV file.
    
    Args:
        filename (str): Name of the CSV file to read. Defaults to 'dataset.csv'.
        
    Returns:
        list: List of dictionaries containing the loaded data with proper type conversion.
        
    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    data = []
    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Convert numeric fields
            row['feature_1'] = float(row['feature_1'])
            row['feature_2'] = float(row['feature_2'])
            row['label'] = int(row['label'])
            data.append(row)
    
    print(f"Dataset loaded from {filename}")
    return data


def split_data(data, train_ratio=0.8):
    """
    Split data into training and evaluation sets.
    
    Args:
        data (list): List of dictionaries containing the data to split.
        train_ratio (float): Ratio of data to use for training. Defaults to 0.8.
        
    Returns:
        tuple: A tuple containing (training_data, evaluation_data) as lists.
    """
    # Shuffle the data
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)
    
    # Calculate split point
    split_point = int(len(shuffled_data) * train_ratio)
    
    training_data = shuffled_data[:split_point]
    evaluation_data = shuffled_data[split_point:]
    
    print(f"Data split: {len(training_data)} training samples, {len(evaluation_data)} evaluation samples")
    
    return training_data, evaluation_data


def train_classifier(training_data):
    """
    Train a Random Forest classifier on the provided training data.
    
    Args:
        training_data (list): List of dictionaries containing training data
                             with 'feature_1', 'feature_2', and 'label' keys.
        
    Returns:
        RandomForestClassifier: Trained classifier model.
        
    Raises:
        ValueError: If training_data is empty.
    """
    if not training_data:
        raise ValueError("Training data cannot be empty")
    
    # Extract features and labels
    X = []
    y = []
    
    for row in training_data:
        # Assuming features are 'feature_1' and 'feature_2'
        features = [row['feature_1'], row['feature_2']]
        X.append(features)
        y.append(row['label'])
    
    X = np.array(X)
    y = np.array(y)
    
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train the classifier
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(X_train, y_train)
    
    # Validate the model
    y_pred = classifier.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    
    print(f"Training completed. Validation accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))
    
    return classifier


def save_classifier(classifier, filename='classifier_model.pkl'):
    """
    Save the trained classifier to a pickle file.
    
    Args:
        classifier: Trained sklearn classifier model to save.
        filename (str): Name of the pickle file to save the model.
                       Defaults to 'classifier_model.pkl'.
    """
    with open(filename, 'wb') as f:
        pickle.dump(classifier, f)
    print(f"Classifier saved to {filename}")


def load_classifier(filename='classifier_model.pkl'):
    """
    Load a trained classifier from a pickle file.
    
    Args:
        filename (str): Name of the pickle file containing the model.
                       Defaults to 'classifier_model.pkl'.
        
    Returns:
        sklearn classifier: Loaded classifier model.
        
    Raises:
        FileNotFoundError: If the model file is not found.
        Exception: If there's an error loading the model.
    """
    try:
        with open(filename, 'rb') as f:
            classifier = pickle.load(f)
        print(f"Classifier loaded from {filename}")
        return classifier
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file {filename} not found. Please train and save a model first.")
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")


def predict(classifier, data):
    """
    Make predictions using the loaded classifier.
    
    Args:
        classifier: Trained sklearn classifier model.
        data (list): List of dictionaries containing data to predict,
                    with 'feature_1' and 'feature_2' keys.
        
    Returns:
        list: List of predictions as integers.
        
    Raises:
        ValueError: If data is empty.
    """
    if not data:
        raise ValueError("Data cannot be empty")
    
    # Extract features
    X = []
    for row in data:
        features = [row['feature_1'], row['feature_2']]
        X.append(features)
    
    predictions = classifier.predict(X)
    
    return predictions.tolist()


def evaluate(data):
    """
    Evaluate the classifier performance on the provided data.
    
    Uses predict() to calculate and return the accuracy of predictions against true labels.
    
    Args:
        data (list): List of dictionaries containing evaluation data with true labels.
                    Must have 'feature_1', 'feature_2', and 'label' keys.
        
    Returns:
        float: Accuracy score as a float between 0 and 1.
        
    Raises:
        FileNotFoundError: If the classifier model file is not found.
        ValueError: If data is empty.
    """
    # Load the classifier
    classifier = load_classifier()
    
    # Get predictions
    predictions = predict(classifier, data)
    
    # Extract true labels
    true_labels = [row['label'] for row in data]
    
    # Calculate accuracy
    correct = sum(1 for true_label, pred in zip(true_labels, predictions) if true_label == pred)
    accuracy = correct / len(true_labels)
    
    print(f"Evaluation accuracy: {accuracy:.4f}")
    
    return accuracy


def main():
    """
    Main method that executes the complete machine learning pipeline.
    
    Executes all steps in order:
    1. Generate the dataset
    2. Store it in CSV form
    3. Read this CSV
    4. Take 80% data for training and save 20% for evaluation
    5. Train a classifier model
    6. Download (save) the model
    7. Load the model
    8. Run it on the evaluation set
    9. Give model accuracy
    """
    print("Starting machine learning pipeline...")
    
    # Step 1: Generate the dataset
    print("\n1. Generating dataset...")
    dataset = generate_dataset(num_samples=10000)
    
    # Step 2: Store it in CSV form
    print("\n2. Saving dataset to CSV...")
    save_to_csv(dataset, 'dataset.csv')
    
    # Step 3: Read this CSV
    print("\n3. Reading dataset from CSV...")
    loaded_data = read_from_csv('dataset.csv')
    
    # Step 4: Take 80% data for training and save 20% for evaluation
    print("\n4. Splitting data into training and evaluation sets...")
    training_data, evaluation_data = split_data(loaded_data, train_ratio=0.8)
    
    # Step 5: Train a classifier model
    print("\n5. Training classifier model...")
    classifier = train_classifier(training_data)
    
    # Step 6: Download (save) the model
    print("\n6. Saving the trained model...")
    save_classifier(classifier, 'classifier_model.pkl')
    
    # Step 7: Load the model
    print("\n7. Loading the saved model...")
    loaded_classifier = load_classifier('classifier_model.pkl')
    
    # Step 8: Run it on the evaluation set
    print("\n8. Running model on evaluation set...")
    predictions = predict(loaded_classifier, evaluation_data)
    print(f"Generated {len(predictions)} predictions")
    
    # Step 9: Give model accuracy
    print("\n9. Calculating model accuracy...")
    accuracy = evaluate(evaluation_data)
    
    print(f"\nPipeline completed successfully!")
    print(f"Final model accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
