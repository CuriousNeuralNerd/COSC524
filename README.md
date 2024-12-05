# Agatha Christie Text Classification using BERT

## Project Overview
This repository contains implementations of three transformer models (BERT, DistilBERT, and XLM-RoBERTa) fine-tuned to identify whether a text was written by Agatha Christie. The project was developed as a final assignment for COSC524 Natural Language Processing, comparing model architectures and exploring the impact of cross-validation and Named Entity Recognition (NER) on performance.

## Repository Structure
- `balanced_results/`: Contains test results from experiments using a balanced dataset
- `unbalanced_results/`: Contains test results from experiments using an unbalanced dataset (2:1 ratio favoring non-Christie authors)
- `data/`: Contains the dataset used for training and testing
- `Untitled.ipynb`: Initial implementation without cross-validation
- `new.ipynb`: Implementation with cross-validation
- `new_with_NER.ipynb`: Final implementation incorporating both cross-validation and Named Entity Recognition

## Model Architectures
1. **BERT** (bert-base-uncased)
   - Full BERT architecture
   - 12 transformer layers
   - 768 hidden dimensions
   - 12 attention heads

2. **DistilBERT** (distilbert-base-cased)
   - Distilled (Smaller) version of BERT
   - 6 transformer layers
   - 768 hidden dimensions

3. **XLM-RoBERTa** (xlm-roberta-base)
   - Multilingual model
   - 12 transformer layers
   - 768 hidden dimensions
   - Trained on 100 languages

## Model Versions
The project evolved through several iterations:
1. Base Model: Initial fine-tuning implementation
2. Cross-Validation Model: Enhanced version with k-fold cross-validation for better reliability
3. NER-Enhanced Model: Final version incorporating Named Entity Recognition features

## Dataset
The project uses two different dataset configurations:
- Balanced Dataset: Equal distribution of Christie and non-Christie texts
- Unbalanced Dataset: 2:1 ratio of non-Christie to Christie texts, reflecting a more realistic scenario

## Experiments
The repository includes multiple experimental notebooks showcasing the model's evolution:
- Basic implementation without validation splits
- Implementation with cross-validation to ensure model reliability
- Advanced implementation combining cross-validation with Named Entity Recognition to capture authorial patterns in entity usage

## Results
Results for multple expirements from both balanced and unbalanced datasets are stored in their respective directories:
- `balanced_results/`: Performance metrics on equally distributed data
- `unbalanced_results/`: Performance metrics on the 2:1 distribution scenario


## Model Evolution & Results

### 1. Base Implementation (Untitled.ipynb)
Simple train/test split without cross-validation or NER
- BERT Test Results:
  - Accuracy: 78.91%
  - Precision: 54.00%
  - Recall: 77.97%
  - F1 Score: 63.80%

- DistilBERT Test Results:
  - Accuracy: 68.86%
  - Precision: 42.23%
  - Recall: 83.30%
  - F1 Score: 56.05%

- XLM-RoBERTa Test Results:
  - Accuracy: 77.84%
  - Precision: 52.36%
  - Recall: 78.31%
  - F1 Score: 62.76%

### 2. Cross-Validation Implementation (new.ipynb)
Added 3-fold cross-validation to improve reliability
- BERT Cross-Validation Results:
  - Accuracy: 98.65%
  - Precision: 98.62%
  - Recall: 98.75%
  - F1 Score: 98.69%

- DistilBERT Cross-Validation Results:
  - Accuracy: 97.78%
  - Precision: 97.19%
  - Recall: 98.52%
  - F1 Score: 97.85%

- XLM-RoBERTa Cross-Validation Results:
  - Accuracy: 97.68%
  - Precision: 97.94%
  - Recall: 97.49%
  - F1 Score: 97.71%

### 3. NER + Cross-Validation Implementation (new_with_NER.ipynb)
Added Named Entity Recognition
- BERT Final Results:
  - Accuracy: 98.66%
  - Precision: 98.63%
  - Recall: 98.75%
  - F1 Score: 98.69%

- DistilBERT Final Results:
  - Accuracy: 97.78%
  - Precision: 97.19%
  - Recall: 98.52%
  - F1 Score: 97.85%

- XLM-RoBERTa Final Results:
  - Accuracy: 97.68%
  - Precision: 97.94%
  - Recall: 97.49%
  - F1 Score: 97.71%
 
## Technical Details
- Maximum sequence length: 128 tokens
- Training parameters:
  - Learning rate: 2e-5
  - Batch size: 16 (training), 32 (evaluation)
  - Training epochs: 4
  - Weight decay: 0.01
- 3-fold cross-validation


## Technical Requirements
- Python 3.10
- PyTorch
- Transformers
- scikit-learn
- pandas
- numpy
- matplotlib

## License
This project is part of academic coursework for COSC524 NLP.
