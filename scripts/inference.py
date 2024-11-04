# Script for Performing Inference using Fine-Tuned NER Model

"""
This script loads a fine-tuned BERT model for Named Entity Recognition (NER) and performs inference on new input text.
It uses the trained model to recognize mountain names from input sentences.
You can choose between two approaches: standard inference using token classification or using a pre-built pipeline for NER.
"""

# Import necessary libraries
import torch
import re
import argparse
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Load tokenizer and model
model_path = "Lizrek/bert-base-mountain-NER"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Define label mapping
label_map = {0: "O", 1: "B-MOUNTAIN_NAME", 2: "I-MOUNTAIN_NAME"}

# Function to perform inference
def predict_mountain_names(text):
    # Tokenize input text
    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    tokens_to_device = tokens.to(device)

    # Get model predictions
    with torch.no_grad():
        outputs = model(**tokens_to_device)
    predictions = torch.argmax(outputs.logits, dim=2)

    # Extract labels and words
    input_ids = tokens["input_ids"][0]
    tokens_text = tokenizer.convert_ids_to_tokens(input_ids)
    word_ids = tokens.word_ids(batch_index=0)
    labels = [label_map[predictions[0][i].item()] for i in range(len(input_ids))]

    # Filter out special tokens ([CLS], [SEP])
    filtered_tokens = []
    filtered_labels = []
    for token, label, word_id in zip(tokens_text, labels, word_ids):
        if word_id is not None: # Ignore special tokens
            filtered_tokens.append(token)
            filtered_labels.append(label)

    highlighted_text = []
    current_entity = []
    current_label = None

    for token, label in zip(filtered_tokens, filtered_labels):
        # Handle subword tokens (e.g., ## for BERT)
        if token.startswith("##"):
            token=token[2:]
            if current_entity:
                current_entity[-1] += token
            else:
                highlighted_text.append(token)
            continue

        if label.startswith("B-"):
            # Save the current entity if it exists
            if current_entity:
                highlighted_text.append(f"[{current_label}: {' '.join(current_entity)}]")
                current_entity = []
            current_entity.append(token)
            current_label = label[2:]
        elif label.startswith("I-") and current_label is not None:
            current_entity.append(token)
        else:
            # End of the current entity
            if current_entity:
                highlighted_text.append(f"[{current_label}: {' '.join(current_entity)}]")
                current_entity = []
            current_label = None
            highlighted_text.append(token)
    
    # Save the final entity if it exists
    if current_entity:
        highlighted_text.append(f"[{current_label}: {' '.join(current_entity)}]")
    
    return " ".join(highlighted_text)

# Main function to allow for command-line execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform NER inference using a fine-tuned model.")
    parser.add_argument("text", type=str, help="Input text to perform NER on.")
    parser.add_argument("--method", type=str, choices=["custom", "pipeline"], default="custom",
                        help="Choose the inference method: 'custom' for manual processing or 'pipeline' for using the NER pipeline.")
    args = parser.parse_args()

    # Get the input text from command line
    input_text = args.text

    if args.method == "custom":
        highlighted_output = predict_mountain_names(input_text)
        print(f"Input: {input_text}")
        print(f"Output: {highlighted_output}")
    elif args.method == "pipeline":
        # Use the Transformers pipeline for NER
        nlp = pipeline("ner", model=model, tokenizer=tokenizer, device=0 if device == "cuda" else -1)
        ner_results = nlp(input_text)
        print(f"Input: {input_text}")
        print(f"Entities: {ner_results}")
