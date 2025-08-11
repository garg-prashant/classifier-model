# Machine Learning Pipeline

A complete machine learning pipeline for binary classification using synthetic data.

## Overview

This project implements a machine learning pipeline that generates synthetic datasets, trains a Random Forest classifier, and evaluates model performance. The pipeline includes data generation, CSV I/O operations, model training with scikit-learn, model persistence, and comprehensive evaluation metrics.

## Features

- Synthetic dataset generation with configurable sample size
- CSV data persistence and loading capabilities
- Random Forest classifier implementation
- Train/test data splitting
- Model serialization using pickle
- Performance evaluation with accuracy scores and classification reports
- Complete pipeline automation from data generation to model evaluation

## Setup python3 virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## Installation

pip install -r requirements.txt

## Usage

python train_model.py

## Contributing

Contributions are welcome! Please feel free to submit a pull request.

## License

## Output:
```bash
(.venv) (base) ➜  programming_test python train_model.py
Starting machine learning pipeline...

1. Generating dataset...

2. Saving dataset to CSV...
Dataset saved to dataset.csv

3. Reading dataset from CSV...
Dataset loaded from dataset.csv

4. Splitting data into training and evaluation sets...
Data split: 8000 training samples, 2000 evaluation samples

5. Training classifier model...
Training completed. Validation accuracy: 0.9906

Classification Report:
              precision    recall  f1-score   support

           0       0.99      0.99      0.99       823
           1       0.99      0.99      0.99       777

    accuracy                           0.99      1600
   macro avg       0.99      0.99      0.99      1600
weighted avg       0.99      0.99      0.99      1600


6. Saving the trained model...
Classifier saved to classifier_model.pkl

7. Loading the saved model...
Classifier loaded from classifier_model.pkl

8. Running model on evaluation set...
Generated 2000 predictions

9. Calculating model accuracy...
Classifier loaded from classifier_model.pkl
Evaluation accuracy: 0.9910

Pipeline completed successfully!
Final model accuracy: 0.9910
```