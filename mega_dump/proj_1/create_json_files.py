#!/usr/bin/env python3
"""
Script to create JSON files for each accession code from ENA sample metadata.
"""

import json
import os
import pandas as pd
from ena_sample_extractor import ENASampleExtractor

def create_json_files_for_accessions():
    """Create individual JSON files for each accession code."""
    
    # Read accession codes
    codes = list(pd.read_csv('disease.csv')['Accession #'])
    print(f"Processing {len(codes)} accession codes...")
    
    # Get sample data
    extractor = ENASampleExtractor()
    results = extractor.process_multiple_studies(codes)
    
    # Create output directory for JSON files
    output_dir = "accession_json_files"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Group samples by study accession
    studies_data = {}
    for sample in results["samples"]:
        study_acc = sample.get('study_accession', 'Unknown')
        if study_acc not in studies_data:
            studies_data[study_acc] = []
        studies_data[study_acc].append(sample)
    
    # Create JSON file for each accession code
    for study_acc, samples in studies_data.items():
        # Create the JSON structure with study accession, total samples, and dicts of lists
        json_data = {
            "study_accession": study_acc,
            "total_samples_found": len(samples),
            "alias": [sample.get('sample_alias', 'N/A') for sample in samples],
            "title": [sample.get('sample_title', 'N/A') for sample in samples]
        }
        
        # Save to JSON file
        filename = f"{output_dir}/{study_acc}_metadata.json"
        with open(filename, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"✓ Created JSON file: {filename}")
    
    print(f"\n✓ All JSON files created in '{output_dir}' directory")
    print(f"✓ Total files created: {len(studies_data)}")
    
    # Show example of one JSON file
    if studies_data:
        first_study = list(studies_data.keys())[0]
        example_file = f"{output_dir}/{first_study}_metadata.json"
        print(f"\n=== EXAMPLE JSON STRUCTURE ({example_file}) ===")
        with open(example_file, 'r') as f:
            example_data = json.load(f)
            print(json.dumps(example_data, indent=2))

if __name__ == "__main__":
    create_json_files_for_accessions() 