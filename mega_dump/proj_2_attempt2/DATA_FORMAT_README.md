# PubMedBERT NER Training - Data Format Guide

## Configuration Summary

**Data Format Chosen: CoNLL Format (Single File)**

This is the standard format for NER tasks and is the most efficient approach.

## Why CoNLL Format?

1. **Industry Standard**: Used by major NER datasets (CoNLL-2003, OntoNotes, etc.)
2. **Efficiency**: Single file contains both tokens and labels
3. **Easy to Parse**: Simple tab/space-separated format
4. **Better for Training**: Maintains sentence boundaries naturally
5. **HuggingFace Compatible**: Works seamlessly with Transformers library

## File Format Specification

### Structure
```
<token>\t<BIO_tag>
<token>\t<BIO_tag>
...
[blank line to separate sentences]
<token>\t<BIO_tag>
...
```

### Example (data/train.conll)
```
Patients	O
with	O
diabetes	B-DISEASE
mellitus	I-DISEASE
often	O
present	O
with	O
elevated	O
insulin	B-PROTEIN
resistance	O
.	O

The	O
BRCA1	B-GENE
gene	O
mutation	O
increases	O
risk	O
of	O
breast	B-DISEASE
cancer	I-DISEASE
.	O
```

## BIO Tagging Scheme

- **B-ENTITY**: Beginning of an entity
- **I-ENTITY**: Inside/continuation of an entity
- **O**: Outside any entity (not an entity)

### Common Biomedical Entity Types
- `B-DISEASE` / `I-DISEASE`: Disease names
- `B-DRUG` / `I-DRUG`: Medication/drug names
- `B-PROTEIN` / `I-PROTEIN`: Protein names
- `B-GENE` / `I-GENE`: Gene names
- `B-CHEMICAL` / `I-CHEMICAL`: Chemical compounds
- `B-CELL` / `I-CELL`: Cell types
- `B-DNA` / `I-DNA`: DNA sequences
- `B-RNA` / `I-RNA`: RNA sequences
- `B-DOSE` / `I-DOSE`: Medication dosages

## File Organization

```
proj_2_attempt2/
├── trainNER.py          # Main training script
├── data/
│   ├── train.conll      # Training data (REQUIRED)
│   ├── val.conll        # Validation data (optional but recommended)
│   └── test.conll       # Test data (optional)
└── DATA_FORMAT_README.md
```

## How to Prepare Your Data

### Option 1: Manual Annotation
1. Create a `.conll` file for each split (train/val/test)
2. For each sentence:
   - Write one token per line with its BIO tag (tab-separated)
   - Add a blank line after each sentence
   - Make sure multi-word entities use B- for first word, I- for rest

### Option 2: Convert from Other Formats

If you have data in a different format, you can convert it:

#### From JSON format:
```python
# Input: [{"text": "...", "entities": [{"start": 0, "end": 5, "label": "DISEASE"}]}]

def json_to_conll(json_data, output_file):
    with open(output_file, 'w') as f:
        for example in json_data:
            text = example['text']
            entities = example['entities']
            tokens = text.split()  # or use better tokenization

            # Create BIO tags
            bio_tags = ['O'] * len(tokens)
            for entity in entities:
                # Map entity spans to token indices and assign B-/I- tags
                # ... (implement span-to-token mapping)
                pass

            # Write to file
            for token, tag in zip(tokens, bio_tags):
                f.write(f"{token}\t{tag}\n")
            f.write("\n")  # Blank line between sentences
```

#### From separate token/tag files:
```python
# If you have tokens.txt and tags.txt

def separate_to_conll(tokens_file, tags_file, output_file):
    with open(tokens_file) as tf, open(tags_file) as gf, open(output_file, 'w') as out:
        for token_line, tag_line in zip(tf, gf):
            tokens = token_line.strip().split()
            tags = tag_line.strip().split()

            for token, tag in zip(tokens, tags):
                out.write(f"{token}\t{tag}\n")
            out.write("\n")
```

## Running the Training Script

### 1. Install Dependencies
```bash
pip install transformers datasets seqeval torch numpy
```

### 2. Prepare Your Data
- Place your data files in the `data/` directory
- Use the CoNLL format as shown in examples
- Name files: `train.conll`, `val.conll`, `test.conll`

### 3. Configure Paths in trainNER.py
Edit these lines in `trainNER.py`:
```python
TRAIN_FILE = "data/train.conll"  # Your training data
VAL_FILE = "data/val.conll"      # Your validation data
TEST_FILE = "data/test.conll"    # Your test data
```

### 4. Run Training
```bash
python trainNER.py
```

## Configuration Options

Edit these in `trainNER.py`:

```python
# Model
MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
OUTPUT_DIR = "./pubmedbert_ner_model"

# Hyperparameters
LEARNING_RATE = 5e-5      # Learning rate
BATCH_SIZE = 16           # Batch size (reduce if OOM)
NUM_EPOCHS = 5            # Number of training epochs
WEIGHT_DECAY = 0.01       # Weight decay for regularization
WARMUP_STEPS = 500        # Warmup steps for learning rate scheduler
```

## Output

After training, you'll get:
- Trained model in `./pubmedbert_ner_model/`
- Label mappings in `label_mappings.json`
- Training logs in `./pubmedbert_ner_model/logs/`
- Evaluation metrics (precision, recall, F1) printed to console

## Tips for Best Results

1. **Data Quality**: Ensure consistent BIO tagging
2. **Data Size**: Aim for at least 1000+ annotated sentences for training
3. **Class Balance**: Try to have balanced examples of each entity type
4. **Validation Set**: Always use a validation set to monitor overfitting
5. **Hyperparameter Tuning**: Experiment with learning rate and batch size
6. **Entity Types**: Keep entity types consistent across all data files
