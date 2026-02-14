Frameworks (what you'd actually use)
1. LLM-IE (UT Health Houston, JAMIA Open 2025)

Purpose-built for biomedical IE — NER, attributes, and relation extraction in one pipeline
Has a built-in "Prompt Editor" agent that helps you write extraction prompts
Works with any LLM backend (Ollama, vLLM, HuggingFace, OpenAI)
Benchmarked on clinical datasets: ~70% F1 for entity extraction, ~60% for attributes
github.com/UTHEALTH/LLM-IE
Most relevant to you — it's the biomedical-specific equivalent of LangExtract

2. RELATE (arXiv, Nov 2025)

Three-stage pipeline: extract relations with LLM → embed with SapBERT → map to ontology predicates
Specifically designed for building biomedical KGs from abstracts — directly your use case
94% accuracy@10 on ChemProt benchmark
Handles negation detection (important — "X was NOT associated with Y")
The ontology mapping step would standardize your outputs automatically

3. Google LangExtract (the one you found)

Most polished, best chunking/parallelism, but general-purpose not biomedical-specific