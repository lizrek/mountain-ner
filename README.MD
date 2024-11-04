# Mountain-NER Project

## Overview

The Mountain-NER project is designed to create a Named Entity Recognition (NER) model that can identify and annotate mountain names in text. The project involves data preparation, model training using a pre-trained BERT model, and inference to identify mountain names in new input text. This repository contains the necessary scripts, data, and instructions to replicate and understand the NER pipeline for this task.

## How to Run the Project

### 1. Set Up Environment

Ensure Python 3.8+ is installed and create a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate   # Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Train the Model

Run the `train_model.py` script to fine-tune the NER model:

```bash
python scripts/train_model.py
```

### 3. Perform Inference

Use `inference.py` to test the model on new text:

```bash
python scripts/inference.py "Mount Everest is one of the most well-known peaks in the world."
```

Options for inference method (`--method`):

- `custom`: Custom processing with detailed token handling.
- `pipeline`: Uses the `transformers` NER pipeline.

Example:

```bash
python scripts/inference.py "Kilimanjaro and Mount Fuji are popular destinations." --method pipeline
```

## Results

The fine-tuned model achieves high precision, recall, F1-score, and accuracy in identifying mountain names in various positions within the text.

## Acknowledgements

This project utilizes the [Hugging Face Transformers library](https://huggingface.co/transformers/) for model training and inference. The `seqeval` library is used for evaluation metrics.
