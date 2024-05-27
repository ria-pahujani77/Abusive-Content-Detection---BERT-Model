# Abuse Content Detection using BERT

This project aims to detect abuse content in text using the BERT (Bidirectional Encoder Representations from Transformers) model. The model is fine-tuned for a classification task to categorize text into hate speech, offensive language, or neither.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Data Preprocessing](#data-preprocessing)
- [Model Configuration](#model-configuration)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The objective of this project is to leverage the BERT model to identify and categorize abusive content in text data. The model is trained and evaluated on a dataset containing labeled instances of hate speech, offensive language, and neutral content.

## Dataset

The dataset used for this project is sourced from Kaggle's Hate Speech and Offensive Language Dataset( the csv file). It consists of labeled text data with three classes:
- 0: Hate Speech
- 1: Offensive Language
- 2: Neither

## Installation

To set up this project, clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/your-username/abuse-content-detection.git
cd abuse-content-detection
pip install -r requirements.txt
```

## Data Preprocessing

The preprocessing involves cleaning the text data, tokenizing it using the BERT tokenizer, and converting the tokens into input IDs and attention masks. The data is then split into training and validation sets.

## Model Configuration

The BERT model is configured with the following parameters:

 - Pre-trained BERT model: bert-base-uncased
 - Sequence length: 128 tokens
 - Number of epochs: 20
 - Learning rate: 2e-5
 - Batch size: 32
 - Dropout rate: 0.1
 - BERT is chosen for its state-of-the-art performance in NLP tasks, including text classification.

##Training

The model is trained using the AdamW optimizer with a linear learning rate scheduler. The training process includes regularization techniques such as dropout to prevent overfitting.

## Evaluation

The model's performance is evaluated using accuracy, precision, recall, and F1-score metrics on the validation set.

## Results

The final model achieved the following performance metrics:

 - Accuracy: 91%
 - Precision: 0.90
 - Recall: 0.91
 - F1-score: 0.91

## Usage
To use the model for inference, run the following script:

```bash
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load the pre-trained BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
model.load_state_dict(torch.load('path_to_your_trained_model.pth'))
model.eval()

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True, padding="max_length")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    return predicted_class

# Example usage
text = "Your text here"
print("Predicted class:", predict(text))
```

