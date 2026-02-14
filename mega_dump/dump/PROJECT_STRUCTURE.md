# Research Lab Project Structure

## 📁 **Clean Codebase Overview**

After cleanup, here's what remains for research paper text processing:

### **🔬 Core Text Processing Tools**
- `research_text_cleaner.py` - **Main Llama3 text cleaning script**
- `requirements_text_cleaner.txt` - Dependencies for text processing
- `README_text_cleaner.md` - Comprehensive documentation

### **📊 Project 1: ENA Sample Processing**
`proj_1/` - Genomic data processing
- `ENA_Sample_Processing_Complete.ipynb` - Main analysis notebook
- `ena_sample_extractor.py` - Sample data extraction
- `accession_json_files/` - 17 metadata JSON files
- `disease.csv`, `ena_samples_metadata.csv` - Dataset files

### **🧬 Project 2: Knowledge Graph & Text Analysis**
`proj_2/` - Advanced biomedical text analysis
- `research_content_cleaned_20250824_222305.json` - **Main research data (2026 articles)**
- `model.safetensors` - **Trained model (438MB)**
- `PHASE2_FINAL_ANALYSIS_REPORT.md` - Research findings
- Analysis scripts:
  - `enhanced_relation_extractor.py` - Entity/relation extraction
  - `comprehensive_kg_analysis.py` - Knowledge graph analysis
  - `quality_validation_analysis.py` - Data quality validation
  - `test_hf_models.py` - Model testing utilities

### **🛠 Environment & Dependencies**
- `labenv/` - Python virtual environment with ML packages
- `Research_Data_Processing_Documentation.txt` - General documentation

## 🎯 **What's Ready for Research Paper Processing**

### **Primary Workflow:**
1. **Input Data**: `proj_2/research_content_cleaned_20250824_222305.json`
   - Contains 2,026 research articles with chunked text
   - Ready for Llama3 processing

2. **Processing Tool**: `research_text_cleaner.py`
   - Llama3 q4km integration
   - Intelligent chunk merging
   - Scientific content preservation
   - Memory-efficient batch processing

3. **Trained Model**: `proj_2/model.safetensors`
   - Custom biomedical model (438MB)
   - Likely trained on research corpus

### **Secondary Analysis Tools:**
- Knowledge graph construction and analysis
- Entity/relation extraction for biomedical text
- Quality validation and metrics
- HuggingFace model integration

## 🚀 **Next Steps**

1. **Fix Llama3 Authentication** - Get HuggingFace access for gated models
2. **Run Text Cleaning** - Process the 2,026 articles
3. **Integrate Custom Model** - Use the trained model.safetensors
4. **Generate Clean Dataset** - Output ready for downstream NLP tasks

## 📈 **Research Value**

This cleaned codebase focuses on:
- ✅ **Text Processing**: Llama3-powered cleaning of research articles
- ✅ **Biomedical NLP**: Custom trained models for domain-specific tasks
- ✅ **Knowledge Extraction**: Entity and relation extraction from literature
- ✅ **Quality Assurance**: Validation and analysis tools
- ✅ **Scalability**: Memory-efficient processing of large corpora

**Total Research Articles**: 2,026 articles ready for processing
**Custom Model**: 438MB trained model available
**Environment**: Complete ML stack with transformers, torch, etc.
