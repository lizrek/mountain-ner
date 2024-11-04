# Script for fine-Tuning a Named Entity Recognition (NER) model

"""
This script fine-tunes a pre-trained BERT model on the annotated dataset with mountain names using BIO tagging.
It includes tokenization, model definition, training, and evaluation steps.
"""

# Import necessary libraries
import json
import torch
import evaluate
from transformers import (AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments, DataCollatorForTokenClassification, AutoConfig)
from sklearn.model_selection import train_test_split

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load and prepare the annotated data
# Load the JSON file with annotated data
annotated_data_path = "data/processed_data/annotated_mountain_sentences_bio.json"
with open(annotated_data_path, 'r', encoding='utf-8') as file:
    annotated_data = json.load(file)

# Extract texts and labels from the annotated data
texts = [item['tokens'] for item in annotated_data]
labels = [item['labels'] for item in annotated_data]

# Convert labels from strings to IDs
label2id = {"O": 0, "B-MOUNTAIN_NAME": 1, "I-MOUNTAIN_NAME": 2}
labels = [[label2id[label] for label in label_seq] for label_seq in labels]

# Split data into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")

# Tokenize the texts and align labels to tokens
def tokenize_and_align_labels(texts, labels):
    tokenized_inputs = tokenizer(texts, truncation=True, is_split_into_words=True, padding=True)
    label_all_tokens = True

    labels_aligned = []
    for i, label in enumerate(labels):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx
        labels_aligned.append(label_ids)

    tokenized_inputs["labels"] = labels_aligned
    return tokenized_inputs

train_encodings = tokenize_and_align_labels(train_texts, train_labels)
val_encodings = tokenize_and_align_labels(val_texts, val_labels)

# Convert encodings to dataset format
class NERDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.input_ids = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]
        self.labels = encodings["labels"]   

    def __getitem__(self, idx	):
        item = {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[idx], dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }
        return item

    def __len__(self):
        return len(self.input_ids)
    
train_dataset = NERDataset(train_encodings)
val_dataset = NERDataset(val_encodings)

# Define model configuration
label_list = ['O', 'B-MOUNTAIN_NAME', 'I-MOUNTAIN_NAME']
num_labels = len(label_list)

config = AutoConfig.from_pretrained(
    "dslim/bert-base-NER",
    num_labels=num_labels,
    id2label={0: "O", 1: "B-MOUNTAIN_NAME", 2: "I-MOUNTAIN_NAME"},
    label2id=label2id
)

# Load pre-trained model for token classification
model = AutoModelForTokenClassification.from_pretrained(
    "dslim/bert-base-NER",
    config=config,
    ignore_mismatched_sizes=True	
)
model.to(device)

# Define data collator
data_collator = DataCollatorForTokenClassification(tokenizer)

# Define metrics for evaluation
metric = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = predictions.argmax(axis=-1)

    true_labels = [
        [label_list[label] for label in label_seq if label != -100]
        for label_seq in labels
    ]
    
    true_predictions = [
        [label_list[pred] for (pred, label) in zip(pred_seq, label_seq) if label != -100]
        for pred_seq, label_seq in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# Define training arguments
training_args = TrainingArguments(
    output_dir="results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="logs",
    logging_steps=10,
    save_steps=500
)

# Initialize trainer
trainer = Trainer(
    model=model,	
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Evaluate the model
results = trainer.evaluate()
print(results)

# Save the fine-tuned model
model.save_pretrained("models/fine-tuned-model")
tokenizer.save_pretrained("models/fine-tuned-model")

print("Model fine-tuning complete and saved.")