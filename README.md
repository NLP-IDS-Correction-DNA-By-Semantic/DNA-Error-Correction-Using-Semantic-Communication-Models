# Semantic Communication Models for DNA Error Correction

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-red)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/-Transformers-yellow)](https://huggingface.co/transformers/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> *Treating DNA as a language: A semantic communication approach to genomic error correction*

## Overview

Errors in DNA sequencing—specifically Insertions, Deletions, and Substitutions (IDS)—pose fundamental challenges in modern genomics. Traditional error correction methods focus on restoring exact bit sequences, which can be computationally inefficient and biologically suboptimal for genomic data.

**Our Innovation:** We reframe DNA error correction as a **semantic communication problem**. By applying Natural Language Processing (NLP) techniques and treating DNA sequences as a biological language, we preserve the **biological meaning** (semantics) rather than just the raw syntax.

### Key Contributions

- **Real Biological Data**: Benchmarked on actual Chimpanzee and Human Mitochondrial genomes, not synthetic sequences
- **Three Novel Architectures**: Systematic comparison of transformer-based, AI-enhanced, and hybrid approaches
- **State-of-the-Art Performance**: Achieved **BER ~0.52** with our novel Hybrid RRNS + MRS pipeline
- **Semantic Validation**: Leveraged DNABERT embeddings to ensure biological accuracy beyond character-level metrics

---

## Architecture Overview

We developed and evaluated three distinct approaches to DNA error correction:

### 1. **DeepJSOC** (Deep Joint Source-Outer Channel Coding)
*Transformer-based end-to-end learning*

- **Approach**: Adapted text transmission models to genomic sequences using Transformer encoder-decoder architecture
- **Key Innovation**: Tokenized DNA into overlapping k-mers instead of traditional words, enabling the model to learn robust semantic representations
- **Goal**: Learn channel-resilient DNA representations that survive noisy transmission without explicit error correction codes

### 2. **SemAI-DNA Inspired Pipeline**
*AI-enhanced semantic filtering*

- **Approach**: Multi-Read Screening (MRS) that filters noisy DNA reads based on semantic quality
- **Core Metric**: DNABERT cosine similarity to identify the most biologically accurate read among multiple noisy copies
- **Advantage**: Improves sequence integrity even when traditional alignment-based classification fails

### 3. **Hybrid RRNS + MRS Pipeline** 
*Novel integration of coding theory and semantic AI*

- **Approach**: Combines classical error correction (Redundant Residue Number Systems) with AI-based semantic filtering
- **Architecture**: 
  - **Stage 1**: Semantic pre-filtering using MRS to select highest-quality reads
  - **Stage 2**: RRNS decoding for structural error correction
- **Performance**: Achieved lowest Bit-Error Rate (**BER ~0.52** at p=0.02), outperforming all baselines

---

## Experimental Results

We evaluated methods using dual metrics to capture both syntactic and semantic accuracy:

| Method | Best BER (p=0.02) | Key Observation |
|--------|-------------------|-----------------|
| **RRNS + MSA** | 0.63 | Standard Multiple Sequence Alignment; struggles with high noise |
| **RRNS + Marker** | 0.66 | Good synchronization but higher overhead |
| **Hybrid RRNS + MRS** | **0.52** | **Best performance**: Semantic filtering significantly cleans input before decoding |

### Evaluation Metrics

1. **Levenshtein Distance**: Character-level accuracy (syntactic)
2. **DNABERT Similarity**: Semantic accuracy using pre-trained genomic language models

*Full experimental results, ablation studies, and visualization plots are available in the [Final Report](docs/Final_Report.pdf).*

---

## Project Structure
```
├── docs/
│   ├── Final_Report.pdf      # Complete methodology and results
│   ├── Survey_Report.pdf     # Literature review (12 papers)
│   └── Proposal.pdf          # Initial hypothesis and timeline
├── src/
│   ├── deepjsoc/             # Transformer-based JSOC implementation
│   ├── semai/                # DNABERT semantic filtering & classification
│   └── rrns_hybrid/          # RRNS implementation & Hybrid Pipeline
├── data/                     # Sample datasets (Chimp & Human mtDNA)
├── experiments/              # Training scripts and configuration files
├── results/                  # Output logs, plots, and checkpoints
├── requirements.txt          # Python dependencies
└── README.md
```

---

## Getting Started

### Prerequisites
```bash
pip install torch transformers biopython scikit-learn matplotlib seaborn
pip install dnabert  # For semantic similarity calculations
```

### Quick Example
```python
from src.rrns_hybrid import HybridPipeline
from src.semai import DNABERTFilter

# Initialize hybrid pipeline
pipeline = HybridPipeline(
    filter_model='DNABERT',
    rrns_config={'moduli': [7, 11, 13]}
)

# Process noisy DNA reads
corrected_sequence = pipeline.correct(noisy_reads, reference_genome)
```

---

## Background & Motivation

### Why Semantic Communication for DNA?

Traditional error correction treats all bit errors equally. However, in biology:
- Some mutations are **silent** (synonymous substitutions)
- Others are **catastrophic** (frameshift mutations)
- **Context matters**: The same error can have different biological impacts

By treating DNA as a language and leveraging pre-trained genomic language models (DNABERT), we can:
1. **Prioritize semantically important regions**
2. **Tolerate biologically irrelevant errors**
3. **Improve efficiency** by focusing computational resources where they matter most

---

## Reports

- **[Final-Report](docs/Final-Report.pdf)** – Complete methodology, results, and discussion
- **[Survey Report](docs/Survey-Report.pdf)** – Literature review covering 12 seminal papers in semantic communication
- **[Project Proposal](docs/Project-Proposal.pdf)** – Initial problem formulation and research timeline

---

## Team

This project was developed as part of advanced coursework at **Bilkent University** by:

| Name | Contribution |
|------|-------------|
| **Sefa Emre Kavgacı** | Hybrid RRNS Pipeline & BER Analysis |
| **Betül Doğrul** | SemAI-DNA Pipeline & Classification |
| **Eslim Rana Emiroğlu** | DeepJSOC Adaptation & Evaluation |

---

## Future Directions

- [ ] Extend to long-read sequencing technologies (Oxford Nanopore, PacBio)
- [ ] Integrate attention visualization to understand which genomic regions the model prioritizes
- [ ] Benchmark on larger datasets (e.g., human chromosome sequences)
- [ ] Explore applications in compressed genomic storage and transmission