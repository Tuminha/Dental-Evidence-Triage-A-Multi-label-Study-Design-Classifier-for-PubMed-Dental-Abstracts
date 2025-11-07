# 🦷 Dental Evidence Triage — A Multi-label Study-Design Classifier for PubMed Dental Abstracts

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Framework](https://img.shields.io/badge/Framework-PyTorch-red.svg)
![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow.svg)
![Status](https://img.shields.io/badge/Status-Active-green.svg)

**Instantly classify dental research abstracts by study design using transformer-based multi-label classification**

[🎯 Overview](#-project-overview) • [📊 Results](#-results) • [🚀 Quick-Start](#-quickstart) • [📓 Notebooks](#-notebooks)

</div>

---

## 👨‍💻 Author

<div align="center">

**Francisco Teixeira Barbosa**

[![GitHub](https://img.shields.io/badge/GitHub-Tuminha-black?style=flat&logo=github)](https://github.com/Tuminha)
[![Kaggle](https://img.shields.io/badge/Kaggle-Profile-20BEFF?style=flat&logo=kaggle&logoColor=white)](https://www.kaggle.com/franciscotbarbosa)
[![Email](https://img.shields.io/badge/Email-cisco%40periospot.com-blue?style=flat&logo=gmail)](mailto:cisco@periospot.com)
[![Twitter](https://img.shields.io/badge/Twitter-cisco__research-1DA1F2?style=flat&logo=twitter)](https://twitter.com/cisco_research)

*Learning Machine Learning through CodeCademy • Building AI solutions step by step*

</div>

---

## 🎯 Project Overview

**Problem**

When scanning literature, the first question is *"what kind of evidence is this?"* (Systematic review? RCT? Case report?). That decision is slow if you must open PDFs.

**Solution**

A compact transformer that reads **title + abstract** and predicts **study design**:

`[SystematicReview, MetaAnalysis, RCT, ClinicalTrial, Cohort, CaseControl, CaseReport, InVitro, Animal, Human]`

**What this repo contains**

- A reproducible pipeline to **ingest PubMed**, **normalize records**, **map Publication Types to labels**, and **split** by year.
- A **multi-label classifier** (DistilBERT) trained on those labels.
- A **Hugging Face push** notebook + a **paste-an-abstract** inference demo.

### 🎓 Learning Objectives

- Master end-to-end NLP pipeline: data acquisition → labeling → training → deployment
- Understand multi-label classification with transformers
- Practice temporal splits to prevent data leakage
- Deploy models to Hugging Face Hub with proper documentation
- Build assistive AI tools for clinical research triage

### 🏆 Key Achievements

- [x] Reproducible PubMed ingestion pipeline with rate-limit handling
- [x] Multi-label mapping from Publication Types to canonical study designs
- [x] Temporal train/val/test splits (2018-2021 / 2022-2023 / 2024-2025)
- [ ] DistilBERT classifier with micro-F1 ≥ 0.75 on common labels
- [ ] Hugging Face Hub deployment with inference widget
- [ ] Error analysis and threshold optimization

---

## 📊 Dataset / Domain

- **Source:** PubMed E-utilities (NCBI)
- **Scope:** Dental research (periodontology, implants, endodontics, orthodontics, oral surgery)
- **Time Range:** 2018–2025 (configurable)
- **Target:** Multi-label study design classification
- **Labels:** 10 canonical categories (SystematicReview, MetaAnalysis, RCT, ClinicalTrial, Cohort, CaseControl, CaseReport, InVitro, Animal, Human)

See [DATACARD.md](DATACARD.md) for detailed data documentation.

---

## 🚀 Quickstart

### Prerequisites

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Environment Setup

Recommended for NCBI rate limits:

```bash
export NCBI_EMAIL="you@example.com"
export NCBI_API_KEY="your_ncbi_key"
```

Get your API key at: https://www.ncbi.nlm.nih.gov/account/settings/

### Usage

Work through notebooks in order (01 → 07):

```bash
jupyter notebook notebooks/
```

1. `01_ingest_pubmed.ipynb` — Fetch raw XML from PubMed
2. `02_normalize_and_label.ipynb` — Parse and label records
3. `03_eda_and_splits.ipynb` — Explore data and create temporal splits
4. `04_train_distilbert_multilabel.ipynb` — Train classifier
5. `05_eval_and_error_analysis.ipynb` — Evaluate and analyze errors
6. `06_push_to_huggingface.ipynb` — Deploy to Hugging Face Hub
7. `07_inference_demo.ipynb` — Interactive inference demo

---

## 📓 Notebooks

### Phase 1: Data Acquisition ✅

**01 - Ingest PubMed Records**
- Query construction with dental MeSH terms
- Batch retrieval with rate limiting
- XML storage for reproducibility

### Phase 2: Data Preparation ✅

**02 - Normalize and Label**
- MEDLINE XML parsing
- Publication Type → label mapping
- Keyword-based backfill for gaps

**03 - EDA and Temporal Splits**
- Class balance analysis
- Label co-occurrence patterns
- Temporal split strategy (prevent leakage)

### Phase 3: Model Training 🔄

**04 - Train DistilBERT Multi-label**
- Tokenization and encoding
- BCEWithLogits loss
- Threshold optimization on validation set

**05 - Evaluation and Error Analysis**
- Per-label precision/recall/F1
- PR curves and threshold tuning
- Qualitative error inspection

### Phase 4: Deployment 📦

**06 - Push to Hugging Face Hub**
- Model card generation
- Hub upload with metadata
- Inference widget configuration

**07 - Inference Demo**
- Interactive Gradio interface
- PMID-based abstract fetching
- Real-time predictions

---

## 🏆 Results

*Coming soon after model training*

Expected targets:
- **Micro-F1:** ≥ 0.75 on common labels (SR/Meta/RCT/CaseReport)
- **Macro-F1:** Lower due to rare labels
- **Per-label AP:** Varies by class prevalence

### 📌 Business Interpretation

- **For Researchers:** Quickly triage literature by evidence strength
- **For Systematic Reviews:** Pre-filter abstracts before full-text screening
- **For Curriculum:** Generate study-design balanced datasets
- **For Retrieval:** Enhance PubMed searches with predicted study types

---

## 🛠 Technical Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Data Acquisition | NCBI E-utilities API | PubMed metadata retrieval |
| Data Processing | Pandas, NumPy | ETL & feature engineering |
| XML Parsing | lxml | MEDLINE XML parsing |
| Visualization | Matplotlib, Seaborn | EDA & results plotting |
| ML Framework | PyTorch, Transformers | Model training |
| Model | DistilBERT | Multi-label classification |
| Evaluation | scikit-learn, evaluate | Metrics computation |
| Deployment | Hugging Face Hub | Model hosting & inference |
| Demo | Gradio | Interactive UI |
| Validation | Pandera | Schema validation |

---

## 📝 Notes on Ethics & Licensing

- **Data Source:** We use PubMed metadata (titles/abstracts/Publication Types, MeSH). Redistribution as derived data is permitted under NCBI guidelines.
- **Restrictions:** Do not include paywalled full texts. For open access full text, prefer Europe PMC OA and track licenses.
- **Intended Use:** Assistive triage tool, not authoritative evidence grading.
- **Limitations:** Silver labels (derived from Publication Types) may have noise; newer papers may lack complete metadata.

---

## 🚀 Next Steps

- [ ] Complete model training and evaluation
- [ ] Optimize per-label thresholds
- [ ] Add focal loss for class imbalance
- [ ] Extend to non-English abstracts
- [ ] Incorporate full-text when available (Europe PMC)
- [ ] Add quality/bias scoring (separate model)
- [ ] Create Streamlit/Gradio web demo
- [ ] Integrate with Periospot AI knowledge pipeline

---

## 📚 Learning Journey

**Skills Applied:**
- Multi-label classification • Transformer fine-tuning • Temporal data splits • API integration • Schema validation • Model deployment • Error analysis

**Challenges Solved:**
- Handling class imbalance in medical literature
- Mapping noisy Publication Types to canonical labels
- Preventing temporal leakage in train/test splits
- Optimizing thresholds for multi-label predictions

---

## 📄 License

MIT License (see [LICENSE](LICENSE))

---

<div align="center">

**⭐ Star this repo if you found it helpful! ⭐**

*Building AI solutions for evidence-based dentistry, one abstract at a time* 🦷🤖

</div>

