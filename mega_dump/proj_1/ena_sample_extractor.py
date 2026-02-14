#!/usr/bin/env python3
"""
ENA Sample Metadata Extractor

This script retrieves sample information (aliases, titles, accessions) 
for given study accessions using the ENA Portal API.
"""

import requests
import pandas as pd
import sys
from typing import List, Dict, Optional

class ENASampleExtractor:
    """Class to extract sample metadata from ENA using study accessions."""
    
    def __init__(self):
        self.base_url = "https://www.ebi.ac.uk/ena/portal/api/search"
        self.default_fields = [
            "sample_accession",
            "secondary_sample_accession", 
            "sample_alias",
            "sample_title",
            "description",
            "scientific_name",
            "study_accession"
        ]
    
    def get_samples_for_study(self, study_accession: str, 
                            fields: Optional[List[str]] = None,
                            output_format: str = "json") -> Dict:
        """
        Get all samples associated with a study accession.
        
        Args:
            study_accession: Study accession (e.g., 'PRJNA1131598')
            fields: List of fields to retrieve (uses default if None)
            output_format: 'json' or 'tsv'
            
        Returns:
            Dictionary with sample data or error information
        """
        if fields is None:
            fields = self.default_fields
            
        params = {
            "result": "sample",
            "query": f'study_accession="{study_accession}"',
            "fields": ",".join(fields),
            "format": output_format,
            "limit": 0  # Get all results
        }
        
        try:
            print(f"Fetching samples for study: {study_accession}")
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            if output_format == "json":
                data = response.json()
                return {
                    "success": True,
                    "study_accession": study_accession,
                    "sample_count": len(data),
                    "samples": data
                }
            else:  # TSV format
                return {
                    "success": True,
                    "study_accession": study_accession,
                    "data": response.text
                }
                
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": f"API request failed: {str(e)}",
                "study_accession": study_accession
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}",
                "study_accession": study_accession
            }
    
    def process_multiple_studies(self, study_accessions: List[str]) -> Dict:
        """
        Process multiple study accessions and combine results.
        
        Args:
            study_accessions: List of study accessions
            
        Returns:
            Combined results from all studies
        """
        all_samples = []
        failed_studies = []
        
        for study_acc in study_accessions:
            result = self.get_samples_for_study(study_acc)
            
            if result["success"]:
                samples = result["samples"]
                print(f"✓ Found {len(samples)} samples for {study_acc}")
                all_samples.extend(samples)
            else:
                print(f"✗ Failed to get samples for {study_acc}: {result['error']}")
                failed_studies.append({
                    "study_accession": study_acc,
                    "error": result["error"]
                })
        
        return {
            "success": len(failed_studies) == 0,
            "total_samples": len(all_samples),
            "samples": all_samples,
            "failed_studies": failed_studies
        }
    
    def save_to_csv(self, samples_data: List[Dict], filename: str) -> bool:
        """
        Save sample data to CSV file.
        
        Args:
            samples_data: List of sample dictionaries
            filename: Output filename
            
        Returns:
            True if successful, False otherwise
        """
        try:
            df = pd.DataFrame(samples_data)
            df.to_csv(filename, index=False)
            print(f"✓ Saved {len(samples_data)} samples to {filename}")
            return True
        except Exception as e:
            print(f"✗ Failed to save to CSV: {str(e)}")
            return False
    
    def print_summary(self, samples_data: List[Dict]) -> None:
        """Print a summary of the retrieved samples."""
        if not samples_data:
            print("No samples found.")
            return
            
        print(f"\n=== SUMMARY ===")
        print(f"Total samples found: {len(samples_data)}")
        
        # Group by study
        studies = {}
        for sample in samples_data:
            study = sample.get('study_accession', 'Unknown')
            if study not in studies:
                studies[study] = []
            studies[study].append(sample)
        
        print(f"Studies processed: {len(studies)}")
        for study, samples in studies.items():
            print(f"  {study}: {len(samples)} samples")
        
        # Show first few samples as examples
        print(f"\n=== SAMPLE EXAMPLES ===")
        for i, sample in enumerate(samples_data[:3]):
            print(f"\nSample {i+1}:")
            print(f"  Accession: {sample.get('sample_accession', 'N/A')}")
            print(f"  Alias: {sample.get('sample_alias', 'N/A')}")
            print(f"  Title: {sample.get('sample_title', 'N/A')}")
            print(f"  Scientific name: {sample.get('scientific_name', 'N/A')}")


def main():
    """Main function to demonstrate usage."""
    extractor = ENASampleExtractor()
    
    # Example study accessions (replace with your own)
    study_accessions = [
        "PRJNA1131598"
        # Add more study accessions here as needed
        # "PRJNA123456",
        # "PRJEB789012"
    ]
    
    print("ENA Sample Metadata Extractor")
    print("=" * 40)
    
    # Process all studies
    results = extractor.process_multiple_studies(study_accessions)
    
    if results["total_samples"] > 0:
        # Print summary
        extractor.print_summary(results["samples"])
        
        # Save to CSV
        output_file = "ena_samples_metadata.csv"
        extractor.save_to_csv(results["samples"], output_file)
        
        print(f"\n✓ Processing complete!")
        print(f"✓ Results saved to: {output_file}")
        
    else:
        print("No samples were retrieved successfully.")
    
    # Report any failures
    if results["failed_studies"]:
        print(f"\n⚠ Warning: {len(results['failed_studies'])} studies failed:")
        for failed in results["failed_studies"]:
            print(f"  {failed['study_accession']}: {failed['error']}")


if __name__ == "__main__":
    main()