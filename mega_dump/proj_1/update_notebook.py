#!/usr/bin/env python3
"""
Script to update the Jupyter notebook with the new JSON creation code.
"""

import json

# Read the current notebook
with open('main.ipynb', 'r') as f:
    notebook = json.load(f)

# The new code for cell 3
new_cell_code = [
    "import json\n",
    "import os\n",
    "\n",
    "# Create output directory for JSON files\n",
    "output_dir = \"accession_json_files\"\n",
    "if not os.path.exists(output_dir):\n",
    "    os.makedirs(output_dir)\n",
    "\n",
    "# Group samples by study accession\n",
    "studies_data = {}\n",
    "for sample in results[\"samples\"]:\n",
    "    study_acc = sample.get('study_accession', 'Unknown')\n",
    "    if study_acc not in studies_data:\n",
    "        studies_data[study_acc] = []\n",
    "    studies_data[study_acc].append(sample)\n",
    "\n",
    "# Create JSON file for each accession code\n",
    "for study_acc, samples in studies_data.items():\n",
    "    # Create the JSON structure with study accession, total samples, and dicts of lists\n",
    "    json_data = {\n",
    "        \"study_accession\": study_acc,\n",
    "        \"total_samples_found\": len(samples),\n",
    "        \"alias\": [sample.get('sample_alias', 'N/A') for sample in samples],\n",
    "        \"title\": [sample.get('sample_title', 'N/A') for sample in samples]\n",
    "    }\n",
    "    \n",
    "    # Save to JSON file\n",
    "    filename = f\"{output_dir}/{study_acc}_metadata.json\"\n",
    "    with open(filename, 'w') as f:\n",
    "        json.dump(json_data, f, indent=2)\n",
    "    \n",
    "    print(f\"✓ Created JSON file: {filename}\")\n",
    "\n",
    "print(f\"\\n✓ All JSON files created in '{output_dir}' directory\")\n",
    "print(f\"✓ Total files created: {len(studies_data)}\")"
]

# Update cell 3 (index 3 in the cells list)
notebook['cells'][3]['source'] = new_cell_code
notebook['cells'][3]['execution_count'] = None
notebook['cells'][3]['outputs'] = []

# Write the updated notebook
with open('main.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("✓ Updated main.ipynb cell 3 with the new JSON creation code") 