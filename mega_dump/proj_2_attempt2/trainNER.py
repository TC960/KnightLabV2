"""
PubMedBERT NER Training Script
Trains a PubMedBERT model for Named Entity Recognition with custom biotagged data
"""

import os
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
import torch


# ============================================================================
# DATA FORMAT CONFIGURATION
# ============================================================================
# Using CoNLL format: Single file with tokens and BIO tags
# Format per line: <token>\t<BIO_tag>
# Blank lines separate sentences
#
# Example (data/train.conll):
# Patients	O
# with	O
# diabetes	B-DISEASE
# mellitus	I-DISEASE
# showed	O
# elevated	O
# insulin	B-PROTEIN
# levels	O
# .	O
#
# The	O
# BRCA1	B-GENE
# gene	O
# mutation	O
# ...
# ============================================================================


class NERDataLoader:
    """Loads NER data in CoNLL format"""

    def __init__(self, file_path):
        self.file_path = file_path
        self.label_list = []

    def read_conll_file(self):
        """
        Reads a CoNLL format file and returns tokens and NER tags
        Returns: list of dictionaries with 'tokens' and 'ner_tags' keys
        """
        examples = []

        with open(self.file_path, 'r', encoding='utf-8') as f:
            tokens = []
            ner_tags = []

            for line in f:
                line = line.strip()

                # Blank line indicates end of sentence
                if not line:
                    if tokens:
                        examples.append({
                            'tokens': tokens,
                            'ner_tags': ner_tags
                        })
                        tokens = []
                        ner_tags = []
                # Skip comment lines
                elif line.startswith('#'):
                    continue
                else:
                    # Split token and tag (tab or space separated)
                    parts = line.split()
                    if len(parts) >= 2:
                        token = parts[0]
                        tag = parts[1]
                        tokens.append(token)
                        ner_tags.append(tag)

            # Add last sentence if file doesn't end with blank line
            if tokens:
                examples.append({
                    'tokens': tokens,
                    'ner_tags': ner_tags
                })

        return examples

    def get_label_list(self, examples):
        """Extract unique labels from the dataset"""
        labels = set()
        for example in examples:
            labels.update(example['ner_tags'])
        return sorted(list(labels))


def load_data(train_file, val_file=None, test_file=None):
    """
    Load training, validation, and test data

    Args:
        train_file: Path to training data in CoNLL format
        val_file: Path to validation data (optional)
        test_file: Path to test data (optional)

    Returns:
        DatasetDict with train/validation/test splits
    """
    datasets = {}

    # Load training data
    train_loader = NERDataLoader(train_file)
    train_examples = train_loader.read_conll_file()
    datasets['train'] = Dataset.from_dict({
        'tokens': [ex['tokens'] for ex in train_examples],
        'ner_tags': [ex['ner_tags'] for ex in train_examples]
    })

    # Load validation data
    if val_file and os.path.exists(val_file):
        val_loader = NERDataLoader(val_file)
        val_examples = val_loader.read_conll_file()
        datasets['validation'] = Dataset.from_dict({
            'tokens': [ex['tokens'] for ex in val_examples],
            'ner_tags': [ex['ner_tags'] for ex in val_examples]
        })

    # Load test data
    if test_file and os.path.exists(test_file):
        test_loader = NERDataLoader(test_file)
        test_examples = test_loader.read_conll_file()
        datasets['test'] = Dataset.from_dict({
            'tokens': [ex['tokens'] for ex in test_examples],
            'ner_tags': [ex['ner_tags'] for ex in test_examples]
        })

    # Get label list from training data
    label_list = train_loader.get_label_list(train_examples)

    return DatasetDict(datasets), label_list


def tokenize_and_align_labels(examples, tokenizer, label_to_id):
    """
    Tokenize inputs and align labels with tokenized inputs
    Handles subword tokenization by assigning labels appropriately
    """
    tokenized_inputs = tokenizer(
        examples['tokens'],
        truncation=True,
        is_split_into_words=True,
        max_length=512
    )

    labels = []
    for i, label in enumerate(examples['ner_tags']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            # Special tokens get label -100 (ignored in loss)
            if word_idx is None:
                label_ids.append(-100)
            # First token of a word gets the label
            elif word_idx != previous_word_idx:
                label_ids.append(label_to_id[label[word_idx]])
            # Subsequent tokens of the same word get -100 (or the same label)
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs['labels'] = labels
    return tokenized_inputs


def compute_metrics(eval_pred):
    """Compute metrics for NER using seqeval"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [id_to_label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id_to_label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    # Calculate metrics
    results = {
        'precision': precision_score(true_labels, true_predictions),
        'recall': recall_score(true_labels, true_predictions),
        'f1': f1_score(true_labels, true_predictions),
    }

    return results


# ============================================================================
# MAIN TRAINING CONFIGURATION
# ============================================================================

if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------

    # Data paths (REPLACE WITH YOUR DATA PATHS)
    TRAIN_FILE = "data/train.conll"  # Path to training data
    VAL_FILE = "data/val.conll"      # Path to validation data (optional)
    TEST_FILE = "data/test.conll"    # Path to test data (optional)

    # Model configuration
    MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    OUTPUT_DIR = "./pubmedbert_ner_model"

    # Training hyperparameters
    LEARNING_RATE = 5e-5
    BATCH_SIZE = 16
    NUM_EPOCHS = 5
    WEIGHT_DECAY = 0.01
    WARMUP_STEPS = 500

    # -------------------------------------------------------------------------
    # Load Data
    # -------------------------------------------------------------------------

    print("Loading data...")
    dataset, label_list = load_data(TRAIN_FILE, VAL_FILE, TEST_FILE)

    print(f"\nFound {len(label_list)} unique labels: {label_list}")
    print(f"Training examples: {len(dataset['train'])}")
    if 'validation' in dataset:
        print(f"Validation examples: {len(dataset['validation'])}")
    if 'test' in dataset:
        print(f"Test examples: {len(dataset['test'])}")

    # Create label mappings
    label_to_id = {label: i for i, label in enumerate(label_list)}
    id_to_label = {i: label for label, i in label_to_id.items()}

    # -------------------------------------------------------------------------
    # Load Tokenizer and Model
    # -------------------------------------------------------------------------

    print(f"\nLoading PubMedBERT model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_list),
        id2label=id_to_label,
        label2id=label_to_id
    )

    # -------------------------------------------------------------------------
    # Tokenize Dataset
    # -------------------------------------------------------------------------

    print("\nTokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_and_align_labels(examples, tokenizer, label_to_id),
        batched=True,
        remove_columns=dataset['train'].column_names
    )

    # -------------------------------------------------------------------------
    # Training Setup
    # -------------------------------------------------------------------------

    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=WARMUP_STEPS,
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        push_to_hub=False,
        report_to="tensorboard"
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset.get('validation', None),
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # -------------------------------------------------------------------------
    # Train Model
    # -------------------------------------------------------------------------

    print("\n" + "="*80)
    print("Starting training...")
    print("="*80 + "\n")

    trainer.train()

    # -------------------------------------------------------------------------
    # Evaluate Model
    # -------------------------------------------------------------------------

    if 'validation' in tokenized_dataset:
        print("\n" + "="*80)
        print("Validation Results:")
        print("="*80)
        val_results = trainer.evaluate()
        print(val_results)

    if 'test' in tokenized_dataset:
        print("\n" + "="*80)
        print("Test Results:")
        print("="*80)
        test_results = trainer.evaluate(tokenized_dataset['test'])
        print(test_results)

        # Get detailed classification report
        predictions = trainer.predict(tokenized_dataset['test'])
        preds = np.argmax(predictions.predictions, axis=2)

        true_predictions = [
            [id_to_label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(preds, predictions.label_ids)
        ]
        true_labels = [
            [id_to_label[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(preds, predictions.label_ids)
        ]

        print("\nDetailed Classification Report:")
        print(classification_report(true_labels, true_predictions))

    # -------------------------------------------------------------------------
    # Save Model
    # -------------------------------------------------------------------------

    print(f"\nSaving model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Save label mappings
    import json
    with open(f"{OUTPUT_DIR}/label_mappings.json", 'w') as f:
        json.dump({
            'label_to_id': label_to_id,
            'id_to_label': id_to_label,
            'label_list': label_list
        }, f, indent=2)

    print("\nTraining complete!")
    print(f"Model saved to: {OUTPUT_DIR}")
