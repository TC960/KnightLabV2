#!/usr/bin/env python3
"""
Research Article Text Cleaner using Llama3
==========================================

This script processes JSON files containing research article chunks and outputs cleaned text
using Llama3 q4km for intelligent text cleaning and merging.

Author: AI Assistant
Date: 2025
"""

import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import gc
import psutil
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Third-party imports
try:
    import torch
    from transformers import (
        AutoTokenizer, 
        AutoModelForCausalLM, 
        AutoConfig,  # Added missing import
        BitsAndBytesConfig,
        GenerationConfig
    )
    from huggingface_hub import login
    from tqdm import tqdm
    import numpy as np
    
except ImportError as e:
    print(f"Error: Missing required packages. Please install: {e}")
    print("Run: pip install torch transformers tqdm numpy accelerate bitsandbytes huggingface-hub")
    sys.exit(1)


@dataclass
class CleaningConfig:
    """Configuration for the text cleaning process."""
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    max_chunk_length: int = 2048
    batch_size: int = 4
    max_memory_gb: float = 16.0
    device: str = "auto"
    quantization: bool = True
    quantization_type: str = "4bit"
    temperature: float = 0.1
    max_new_tokens: int = 1024
    num_threads: int = 2
    checkpoint_interval: int = 10


class MemoryMonitor:
    """Monitor system memory usage."""
    
    def __init__(self, max_memory_gb: float = 16.0):
        self.max_memory_gb = max_memory_gb
        self.max_memory_bytes = max_memory_gb * 1024**3
    
    def get_memory_usage(self) -> Tuple[float, float]:
        """Get current memory usage in GB."""
        process = psutil.Process()
        memory_info = process.memory_info()
        system_memory = psutil.virtual_memory()
        
        process_gb = memory_info.rss / 1024**3
        system_gb = system_memory.used / 1024**3
        
        return process_gb, system_gb
    
    def is_memory_available(self) -> bool:
        """Check if sufficient memory is available."""
        process_gb, system_gb = self.get_memory_usage()
        return process_gb < self.max_memory_gb * 0.8


class TextCleaner:
    """Main text cleaning class using Llama3."""
    
    def __init__(self, config: CleaningConfig):
        self.config = config
        self.memory_monitor = MemoryMonitor(config.max_memory_gb)
        self.model = None
        self.tokenizer = None
        self.device = None
        self.lock = threading.Lock()
        
        # Setup logging
        self.setup_logging()
        
        # Load model and tokenizer
        self.load_model()
    
    def setup_logging(self):
        """Setup comprehensive logging."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"text_cleaning_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Text cleaning session started: {timestamp}")
        self.logger.info(f"Configuration: {self.config}")
    
    def load_model(self):
        """Load cached model without downloading."""
        try:
            self.logger.info(f"Loading cached model: {self.config.model_name}")
            
            # Determine device
            if self.config.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = self.config.device
            
            self.logger.info(f"Using device: {self.device}")
            
            # Load tokenizer
            self.logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.logger.info("✅ Tokenizer loaded successfully")
            
            # Load model with quantization if enabled
            self.logger.info("Loading model...")
            if self.config.quantization:
                self.logger.info("Using 4-bit quantization...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    load_in_4bit=True,
                    trust_remote_code=True
                )
            else:
                self.logger.info("Loading without quantization...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
            
            self.logger.info("✅ Model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            self.logger.info("Setting model to None - will use fallback text cleaning")
            self.model = None
            self.tokenizer = None
    
    def create_cleaning_prompt(self, chunks: List[str], article_id: str) -> str:
        """Create a Llama3-optimized prompt for text cleaning."""
        combined_text = " ".join(chunks)
        
        # Llama3 uses a specific prompt format
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert scientific text editor specializing in biomedical research papers. Your task is to clean and merge research article text chunks while preserving all scientific content.

<|eot_id|><|start_header_id|>user<|end_header_id|>

Clean and merge the following research article chunks into a single, coherent text:

CLEANING REQUIREMENTS:
1. Combine all chunks into one coherent document
2. Fix sentence boundaries between chunks  
3. Remove citation numbers like [1], [2-5], etc.
4. Remove OCR errors and formatting artifacts
5. Standardize biomedical terminology formatting
6. Remove extra whitespace and line breaks
7. Preserve ALL scientific content, entity names, and data
8. Ensure text flows naturally for downstream NLP processing

CRITICAL - PRESERVE:
- ALL scientific terms, species names, gene names, protein names
- Numerical data, percentages, p-values, statistical measures
- Methodology descriptions and experimental procedures
- Statistical results and conclusions
- Chemical formulas and molecular names

REMOVE ONLY:
- Formatting artifacts and OCR errors
- Redundant whitespace
- Citation brackets and numbers
- Broken sentence fragments between chunks

TEXT TO CLEAN:
{combined_text}

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Here is the cleaned and merged text:

"""
        
        return prompt
    
    def clean_with_llama(self, chunks: List[str], article_id: str) -> str:
        """Clean text chunks using Llama3."""
        try:
            prompt = self.create_cleaning_prompt(chunks, article_id)
            
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_chunk_length,
                padding=True
            )
            
            # Move to device if using GPU
            if torch.cuda.is_available():
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                generation_config = GenerationConfig(
                    temperature=self.config.temperature,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=True,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                outputs = self.model.generate(
                    **inputs,
                    generation_config=generation_config
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract cleaned text (remove the prompt part)
            if "Here is the cleaned and merged text:" in response:
                cleaned_text = response.split("Here is the cleaned and merged text:")[-1].strip()
            elif "<|start_header_id|>assistant<|end_header_id|>" in response:
                cleaned_text = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
                if cleaned_text.startswith("Here is the cleaned and merged text:"):
                    cleaned_text = cleaned_text.replace("Here is the cleaned and merged text:", "").strip()
            else:
                # Fallback extraction
                cleaned_text = response.split("TEXT TO CLEAN:")[-1].strip()
            
            # Post-process the cleaned text
            cleaned_text = self.post_process_text(cleaned_text)
            
            return cleaned_text
            
        except Exception as e:
            self.logger.error(f"Error cleaning text for article {article_id}: {e}")
            # Fallback to basic cleaning
            return self.fallback_cleaning(chunks)
    
    def post_process_text(self, text: str) -> str:
        """Post-process the cleaned text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common formatting issues
        text = re.sub(r'\s+([.,;:])', r'\1', text)  # Remove space before punctuation
        text = re.sub(r'([.!?])\s*([a-z])', r'\1 \2', text)  # Ensure space after sentence end
        
        # Remove remaining citation brackets
        text = re.sub(r'\[\d+(?:[-–]\d+)?\]', '', text)
        text = re.sub(r'\[\d+(?:,\s*\d+)*\]', '', text)
        
        # Fix sentence boundaries
        text = re.sub(r'([a-z])([A-Z][a-z])', r'\1. \2', text)
        
        # Clean up multiple periods
        text = re.sub(r'\.{2,}', '.', text)
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def fallback_cleaning(self, chunks: List[str]) -> str:
        """Fallback text cleaning without LLM."""
        combined_text = " ".join(chunks)
        
        # Basic cleaning operations
        text = re.sub(r'\s+', ' ', combined_text)  # Normalize whitespace
        text = re.sub(r'\[\d+(?:[-–]\d+)?\]', '', text)  # Remove citations
        text = re.sub(r'\[\d+(?:,\s*\d+)*\]', '', text)  # Remove multiple citations
        text = re.sub(r'\s+([.,;:])', r'\1', text)  # Fix punctuation spacing
        text = re.sub(r'([.!?])\s*([a-z])', r'\1 \2', text)  # Sentence boundaries
        
        return text.strip()
    
    def process_article(self, article_id: str, chunks: List[str]) -> str:
        """Process a single article."""
        try:
            self.logger.info(f"Processing article {article_id} with {len(chunks)} chunks")
            
            # If no model is loaded, use fallback cleaning
            if self.model is None or self.tokenizer is None:
                self.logger.info(f"No model available, using fallback cleaning for article {article_id}")
                return self.fallback_cleaning(chunks)
            
            # Check memory before processing
            if not self.memory_monitor.is_memory_available():
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                if not self.memory_monitor.is_memory_available():
                    self.logger.warning(f"Low memory, using fallback for article {article_id}")
                    return self.fallback_cleaning(chunks)
            
            # Clean with model
            with self.lock:
                cleaned_text = self.clean_with_llama(chunks, article_id)
            
            self.logger.info(f"Successfully processed article {article_id}")
            return cleaned_text
            
        except Exception as e:
            self.logger.error(f"Error processing article {article_id}: {e}")
            return self.fallback_cleaning(chunks)
    
    def save_checkpoint(self, processed_data: Dict[str, str], checkpoint_file: str):
        """Save processing checkpoint."""
        try:
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Checkpoint saved: {len(processed_data)} articles processed")
        except Exception as e:
            self.logger.error(f"Error saving checkpoint: {e}")
    
    def load_checkpoint(self, checkpoint_file: str) -> Dict[str, str]:
        """Load processing checkpoint."""
        try:
            if os.path.exists(checkpoint_file):
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.logger.info(f"Checkpoint loaded: {len(data)} articles")
                return data
            return {}
        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {e}")
            return {}
    
    def process_json_file(self, input_file: str, output_file: str) -> Dict[str, Any]:
        """Process the entire JSON file."""
        try:
            # Load input data
            self.logger.info(f"Loading input file: {input_file}")
            with open(input_file, 'r', encoding='utf-8') as f:
                input_data = json.load(f)
            
            total_articles = len(input_data)
            self.logger.info(f"Total articles to process: {total_articles}")
            
            # Setup checkpoint
            checkpoint_file = f"{output_file}.checkpoint"
            processed_data = self.load_checkpoint(checkpoint_file)
            
            # Track statistics
            stats = {
                'total_articles': total_articles,
                'processed_articles': len(processed_data),
                'failed_articles': 0,
                'start_time': time.time(),
                'total_chunks_processed': 0,
                'average_chunks_per_article': 0
            }
            
            # Process articles
            with tqdm(total=total_articles, desc="Processing articles") as pbar:
                pbar.update(len(processed_data))  # Update for already processed
                
                for article_id, article_data in input_data.items():
                    if article_id in processed_data:
                        continue  # Skip already processed
                    
                    try:
                        # Extract chunks based on data structure
                        if isinstance(article_data, dict) and 'chunks' in article_data:
                            chunks = article_data['chunks']
                        elif isinstance(article_data, list):
                            chunks = article_data
                        else:
                            self.logger.warning(f"Unexpected data structure for article {article_id}")
                            chunks = [str(article_data)]
                        
                        # Process article
                        cleaned_text = self.process_article(article_id, chunks)
                        processed_data[article_id] = cleaned_text
                        
                        stats['total_chunks_processed'] += len(chunks)
                        
                        # Save checkpoint periodically
                        if len(processed_data) % self.config.checkpoint_interval == 0:
                            self.save_checkpoint(processed_data, checkpoint_file)
                        
                        pbar.update(1)
                        
                        # Memory cleanup
                        if len(processed_data) % 5 == 0:
                            gc.collect()
                        
                    except Exception as e:
                        self.logger.error(f"Failed to process article {article_id}: {e}")
                        stats['failed_articles'] += 1
                        pbar.update(1)
            
            # Final statistics
            stats['processed_articles'] = len(processed_data)
            stats['end_time'] = time.time()
            stats['total_time'] = stats['end_time'] - stats['start_time']
            stats['average_chunks_per_article'] = (
                stats['total_chunks_processed'] / stats['processed_articles'] 
                if stats['processed_articles'] > 0 else 0
            )
            
            # Save final output
            self.logger.info(f"Saving final output to: {output_file}")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, indent=2, ensure_ascii=False)
            
            # Save statistics
            stats_file = f"{output_file}.stats.json"
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2)
            
            # Cleanup checkpoint
            if os.path.exists(checkpoint_file):
                os.remove(checkpoint_file)
            
            self.logger.info("Processing completed successfully")
            self.logger.info(f"Statistics: {stats}")
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error processing file: {e}")
            raise
    
    def cleanup(self):
        """Cleanup resources."""
        try:
            if self.model is not None:
                del self.model
            if self.tokenizer is not None:
                del self.tokenizer
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.logger.info("Cleanup completed")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean research article text using cached models")
    parser.add_argument("input_file", help="Input JSON file path")
    parser.add_argument("output_file", help="Output JSON file path")
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct", 
                       help="Model name (default: meta-llama/Meta-Llama-3-8B-Instruct)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--max-memory", type=float, default=16.0, 
                       help="Maximum memory usage in GB")
    parser.add_argument("--no-quantization", action="store_true", 
                       help="Disable quantization")
    parser.add_argument("--device", default="auto", help="Device to use")
    parser.add_argument("--checkpoint-interval", type=int, default=10,
                       help="Checkpoint save interval")
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found")
        sys.exit(1)
    
    # Create config
    config = CleaningConfig(
        model_name=args.model,
        batch_size=args.batch_size,
        max_memory_gb=args.max_memory,
        quantization=not args.no_quantization,
        device=args.device,
        checkpoint_interval=args.checkpoint_interval
    )
    
    # Initialize cleaner
    cleaner = None
    try:
        print(f"Initializing text cleaner with model: {config.model_name}")
        cleaner = TextCleaner(config)
        
        print(f"Processing file: {args.input_file}")
        stats = cleaner.process_json_file(args.input_file, args.output_file)
        
        print("\n" + "="*50)
        print("PROCESSING COMPLETED")
        print("="*50)
        print(f"Total articles: {stats['total_articles']}")
        print(f"Successfully processed: {stats['processed_articles']}")
        print(f"Failed: {stats['failed_articles']}")
        print(f"Total chunks processed: {stats['total_chunks_processed']}")
        print(f"Average chunks per article: {stats['average_chunks_per_article']:.2f}")
        print(f"Total time: {stats['total_time']:.2f} seconds")
        print(f"Output saved to: {args.output_file}")
        print(f"Statistics saved to: {args.output_file}.stats.json")
        
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        if cleaner:
            cleaner.cleanup()


if __name__ == "__main__":
    main()