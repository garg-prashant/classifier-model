# classifier-model

A machine learning classifier model for text classification tasks.

## Overview

This project implements a text classifier that can categorize text documents into predefined classes. The model uses natural language processing techniques to analyze and classify text data.

## Features

- Text preprocessing and feature extraction
- Multiple classification algorithms support
- Model training and evaluation
- Prediction capabilities for new text data
- Performance metrics and visualization

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