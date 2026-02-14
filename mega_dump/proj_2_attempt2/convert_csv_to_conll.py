"""
Convert BIOtaggedForNER.csv to CoNLL format
Cleans data by removing unnecessary characters: brackets, quotes, commas
"""

import csv
import re


def clean_bio_tags(tag_string):
    """
    Clean BIO tag string by removing brackets, quotes, and parsing into list

    Input: "['O', 'B-DISEASE', 'I-DISEASE', 'O']"
    Output: ['O', 'B-DISEASE', 'I-DISEASE', 'O']
    """
    # Remove brackets
    tag_string = tag_string.replace('[', '').replace(']', '')

    # Remove all quotes (both single and double)
    tag_string = tag_string.replace('"', '').replace("'", '')

    # Split by comma and strip whitespace
    tags = [tag.strip() for tag in tag_string.split(',') if tag.strip()]

    return tags


def clean_tokens(token_list):
    """
    Clean token list by removing unnecessary quotes and empty tokens
    """
    cleaned = []
    for token in token_list:
        # Remove quotes and strip whitespace
        token = token.replace('"', '').replace("'", '').strip()
        if token:  # Only add non-empty tokens
            cleaned.append(token)
    return cleaned


def parse_csv_to_sentences(csv_file_path):
    """
    Parse CSV file and extract sentences with their BIO tags

    CSV structure:
    - Row 1: Headers (skip)
    - Even rows: Tokens (comma-separated)
    - Odd rows: BIO tags (spread across multiple cells)
    - Blank rows separate different entries

    Returns: list of (tokens, tags) tuples
    """
    sentences = []

    with open(csv_file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = list(reader)

    # Skip header row
    i = 1

    while i < len(rows):
        # Check if we have both a token row and a tag row
        if i + 1 < len(rows):
            token_row = rows[i]
            tag_row = rows[i + 1]

            # Clean tokens
            tokens = clean_tokens(token_row)

            # Clean and parse tags - they're spread across all cells in the tag row
            # Join all cells and then clean
            tag_string = ''.join(tag_row)
            tags = clean_bio_tags(tag_string)

            # Only add if we have both tokens and tags
            if tokens and tags:
                # Make sure tokens and tags have the same length
                min_len = min(len(tokens), len(tags))
                tokens = tokens[:min_len]
                tags = tags[:min_len]

                if min_len > 0:  # Only add non-empty sentences
                    sentences.append((tokens, tags))

        # Move to next sentence (skip 2 rows: current token row, tag row, and any blank rows)
        i += 2
        # Skip any blank rows
        while i < len(rows) and all(not cell.strip() for cell in rows[i]):
            i += 1

    return sentences


def write_conll_file(sentences, output_file):
    """
    Write sentences to CoNLL format file

    Format:
    token1\tBIO_tag1
    token2\tBIO_tag2
    ...
    [blank line]
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for tokens, tags in sentences:
            for token, tag in zip(tokens, tags):
                f.write(f"{token}\t{tag}\n")
            f.write("\n")  # Blank line between sentences


def split_data(sentences, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Split sentences into train/validation/test sets
    """
    total = len(sentences)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train = sentences[:train_end]
    val = sentences[train_end:val_end]
    test = sentences[val_end:]

    return train, val, test


def main():
    # Input CSV file
    csv_file = "BIOtaggedForNER.csv"

    # Output CoNLL files
    train_file = "data/train.conll"
    val_file = "data/val.conll"
    test_file = "data/test.conll"

    print("="*80)
    print("Converting BIOtaggedForNER.csv to CoNLL format")
    print("="*80)

    # Parse CSV
    print(f"\nReading CSV file: {csv_file}")
    sentences = parse_csv_to_sentences(csv_file)
    print(f"Found {len(sentences)} sentences")

    # Display first sentence as example
    if sentences:
        print("\nExample sentence (first one):")
        tokens, tags = sentences[0]
        for token, tag in zip(tokens[:10], tags[:10]):  # Show first 10 tokens
            print(f"  {token}\t{tag}")
        if len(tokens) > 10:
            print(f"  ... ({len(tokens) - 10} more tokens)")

    # Collect all unique tags
    all_tags = set()
    for _, tags in sentences:
        all_tags.update(tags)
    print(f"\nFound {len(all_tags)} unique BIO tags:")
    print(f"  {sorted(all_tags)}")

    # Split data
    print(f"\nSplitting data (70% train, 15% val, 15% test)...")
    train, val, test = split_data(sentences, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

    print(f"  Training sentences: {len(train)}")
    print(f"  Validation sentences: {len(val)}")
    print(f"  Test sentences: {len(test)}")

    # Write to CoNLL files
    print(f"\nWriting CoNLL files...")
    write_conll_file(train, train_file)
    print(f"  ✓ {train_file}")

    write_conll_file(val, val_file)
    print(f"  ✓ {val_file}")

    write_conll_file(test, test_file)
    print(f"  ✓ {test_file}")

    print("\n" + "="*80)
    print("Conversion complete!")
    print("="*80)
    print("\nYou can now run: python trainNER.py")


if __name__ == "__main__":
    main()
