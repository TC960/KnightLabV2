#!/usr/bin/env python3
"""Add paper text to annotation JSON files."""

import json
from pathlib import Path

# Process each annotation file
annotations_dir = Path('annotations')
for annotation_file in sorted(annotations_dir.glob('paper_*.json')):
    # Find corresponding text file (e.g., paper_141.json → paper_141_text.txt)
    text_file = annotation_file.parent / f"{annotation_file.stem}_text.txt"

    if not text_file.exists():
        print(f"✗ No text file found for {annotation_file.name}")
        continue

    # Load annotation
    with open(annotation_file, 'r') as f:
        annotation = json.load(f)

    # Read paper text
    with open(text_file, 'r') as f:
        paper_text = f.read()

    # Add paper text
    annotation['paper_text'] = paper_text

    # Write back
    with open(annotation_file, 'w') as f:
        json.dump(annotation, f, indent=2)

    print(f"✓ Added text to {annotation_file.name}")

print("\nDone!")
