# Research Text Cleaner with Llama3 q4km

A powerful Python script that processes JSON files containing research article chunks and outputs cleaned text using Llama3 with 4-bit quantization (q4km) for memory-efficient text cleaning.

## Features

- **Llama3 Integration**: Uses Meta-Llama-3-8B-Instruct with optimized prompts
- **q4km Quantization**: 4-bit quantization with mixed precision for memory efficiency
- **Intelligent Chunk Merging**: Combines article chunks into coherent text
- **Scientific Content Preservation**: Maintains all scientific terms, data, and methodology
- **Memory Management**: Efficient batch processing with memory monitoring
- **Progress Tracking**: Real-time progress bars and comprehensive logging
- **Checkpoint System**: Automatic saving and resuming for large datasets
- **Error Handling**: Robust fallback mechanisms and comprehensive error reporting

## Installation

### Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended, 6-8GB VRAM)
- 16GB+ system RAM

### Install Dependencies

```bash
pip install -r requirements_text_cleaner.txt
```

### Required Packages

- `torch>=2.0.0`
- `transformers>=4.30.0`
- `accelerate>=0.20.0`
- `bitsandbytes>=0.39.0`
- `numpy>=1.21.0`
- `tqdm>=4.65.0`
- `psutil>=5.9.0`

## Usage

### Basic Usage

```bash
python research_text_cleaner.py input.json output.json
```

### Advanced Usage

```bash
# Custom memory limit and batch size
python research_text_cleaner.py input.json output.json --max-memory 4.0 --batch-size 2

# Use different Llama3 model variant
python research_text_cleaner.py input.json output.json --model meta-llama/Meta-Llama-3-70B-Instruct

# Disable quantization (requires more memory)
python research_text_cleaner.py input.json output.json --no-quantization

# Custom checkpoint interval
python research_text_cleaner.py input.json output.json --checkpoint-interval 5
```

### Command Line Options

- `input_file`: Path to input JSON file
- `output_file`: Path to output JSON file
- `--model`: Model name (default: Meta-Llama-3-8B-Instruct)
- `--batch-size`: Batch size for processing (default: 4)
- `--max-memory`: Maximum memory usage in GB (default: 8.0)
- `--no-quantization`: Disable 4-bit quantization
- `--device`: Device to use (auto/cuda/cpu)
- `--checkpoint-interval`: Save checkpoint every N articles (default: 10)

## Input Format

The script accepts JSON files with the following structure:

### Format 1: Object with chunks array
```json
{
  "article_1": {
    "name": "Article Title",
    "url": "https://example.com",
    "chunks": [
      "Abstract\nThe oral microbiota plays an important role...",
      "Methods\nWe collected samples from 50 patients...",
      "Results\nFusobacterium nucleatum was significantly..."
    ]
  }
}
```

### Format 2: Direct chunks array
```json
{
  "article_1": [
    "Introduction\nCancer research has advanced...",
    "This study focuses on the role of...",
    "Our analysis reveals important insights..."
  ]
}
```

## Output Format

The script outputs a JSON file with cleaned, merged text:

```json
{
  "article_1": "The oral microbiota plays an important role in human health. This study examines the relationship between oral bacteria and cancer development. We collected samples from 50 patients with oral squamous cell carcinoma (OSCC). DNA extraction was performed using standard protocols. Fusobacterium nucleatum was significantly increased in cancer tissues (p<0.001). The abundance of Prevotella intermedia was also elevated. Our findings suggest that specific bacterial species may serve as biomarkers for OSCC diagnosis and treatment."
}
```

## Text Cleaning Features

### What Gets Cleaned
- ✅ Citation numbers: `[1]`, `[2-5]`, `[1,3,5]`
- ✅ Extra whitespace and line breaks
- ✅ OCR errors and formatting artifacts
- ✅ Broken sentence boundaries between chunks
- ✅ Inconsistent punctuation spacing

### What Gets Preserved
- ✅ Scientific terms and biomedical terminology
- ✅ Species names, gene names, protein names
- ✅ Numerical data, percentages, p-values
- ✅ Statistical measures and results
- ✅ Methodology descriptions
- ✅ Chemical formulas and molecular names
- ✅ All factual content and conclusions

## Memory Management

The script includes sophisticated memory management:

- **Quantization**: 4-bit quantization reduces memory usage by ~75%
- **Batch Processing**: Processes articles in configurable batches
- **Memory Monitoring**: Real-time memory usage tracking
- **Garbage Collection**: Automatic cleanup between batches
- **Fallback Mode**: Switches to basic cleaning if memory is low

### Memory Requirements

| Configuration | GPU Memory | System RAM |
|---------------|------------|------------|
| Llama3-8B + q4km | 6-8 GB | 16+ GB |
| Llama3-8B (full) | 16+ GB | 32+ GB |
| Llama3-70B + q4km | 40+ GB | 64+ GB |

## Logging and Monitoring

### Log Files
- `text_cleaning_YYYYMMDD_HHMMSS.log`: Detailed processing log
- `output.json.stats.json`: Processing statistics
- `output.json.checkpoint`: Automatic checkpoint (deleted on completion)

### Statistics Tracked
- Total articles processed
- Processing time per article
- Memory usage patterns
- Error rates and types
- Average chunks per article

## Error Handling

The script includes comprehensive error handling:

1. **Model Loading Errors**: Clear error messages for setup issues
2. **Memory Errors**: Automatic fallback to basic cleaning
3. **Processing Errors**: Individual article failures don't stop the batch
4. **File Errors**: Validation of input/output file access
5. **Checkpoint Recovery**: Resume from interruptions

## Performance Optimization

### Tips for Better Performance

1. **GPU Usage**: Use CUDA-compatible GPU for 3-5x speedup
2. **Batch Size**: Adjust based on available memory
3. **Model Selection**: Use 8B model for balance of quality/speed
4. **Quantization**: Keep enabled for memory efficiency
5. **Checkpoints**: Use frequent checkpoints for large datasets

### Typical Processing Speeds

| Configuration | Articles/Hour | Notes |
|---------------|---------------|-------|
| GPU + q4km | 100-200 | Recommended setup |
| GPU (full precision) | 50-100 | Higher quality, more memory |
| CPU only | 10-20 | Slow but works without GPU |

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `--batch-size` to 1-2
   - Lower `--max-memory` setting
   - Use smaller model variant

2. **Model Loading Fails**
   - Check HuggingFace authentication
   - Verify model name spelling
   - Ensure sufficient disk space

3. **Slow Processing**
   - Verify GPU is being used
   - Check system memory usage
   - Consider using smaller model

4. **Poor Cleaning Quality**
   - Try disabling quantization
   - Adjust temperature parameter
   - Use larger model variant

### Getting Help

For issues or questions:
1. Check the log files for detailed error messages
2. Verify your system meets the minimum requirements
3. Try the example script first to test your setup

## Example Usage Script

Run the included example to test your setup:

```bash
python example_usage.py
```

This will:
1. Create sample data
2. Process it with Llama3
3. Show the results
4. Display usage examples

## License and Attribution

This script uses the following open-source components:
- **Transformers**: HuggingFace transformers library
- **Llama3**: Meta's Llama 3 model (subject to Meta's license)
- **BitsAndBytes**: For quantization support

Please ensure you comply with the respective licenses when using this software.

## Contributing

To contribute improvements:
1. Test thoroughly with different data types
2. Add comprehensive error handling
3. Include performance benchmarks
4. Update documentation

---

*Last updated: 2025*
