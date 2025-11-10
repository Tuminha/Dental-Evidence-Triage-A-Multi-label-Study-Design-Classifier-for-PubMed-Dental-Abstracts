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

- [x] **Notebook 01** - Complete PubMed ingestion pipeline (400 XML files, ~76,165 articles)
- [x] **Notebook 02** - Data normalization and multi-label mapping (64,981 labeled articles, 85.3% coverage)
- [ ] Temporal train/val/test splits (2018-2021 / 2022-2023 / 2024-2025)
- [ ] DistilBERT classifier with micro-F1 ≥ 0.75 on common labels
- [ ] Hugging Face Hub deployment with inference widget
- [ ] Error analysis and threshold optimization

---

## 📊 Dataset / Domain

- **Source:** PubMed E-utilities (NCBI)
- **Scope:** Dental research (periodontology, implants, endodontics, orthodontics, oral surgery)
- **Time Range:** 2018–2025 (configurable)
- **Total Articles:** 76,165 articles ingested
- **Labeled Articles:** 64,981 articles (85.3% coverage)
- **Target:** Multi-label study design classification
- **Labels:** 10 canonical categories (SystematicReview, MetaAnalysis, RCT, ClinicalTrial, Cohort, CaseControl, CaseReport, InVitro, Animal, Human)
- **Label Distribution:**
  - Human: 56,804 articles (74.6%)
  - Cohort: 7,035 articles (9.2%)
  - InVitro: 6,786 articles (8.9%)
  - Animal: 6,019 articles (7.9%) - perfect match with MeSH
  - CaseReport: 5,084 articles (6.7%)
  - SystematicReview: 4,128 articles (5.4%)
  - RCT: 3,759 articles (4.9%)
  - CaseControl: 2,460 articles (3.2%)
  - MetaAnalysis: 2,110 articles (2.8%)
  - ClinicalTrial: 503 articles (0.7%)

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

**01 - Ingest PubMed Records** (Complete)
- ✅ Query construction with dental MeSH terms
- ✅ ESearch API integration with pagination
- ✅ Rate limiting and error handling
- ✅ NCBI 10K result limit handling
- ✅ Batch XML retrieval (efetch function)
- ✅ XML storage for reproducibility (400 XML files)
- ✅ 76,165 articles ingested across 2018-2025

### Phase 2: Data Preparation ✅

**02 - Normalize and Label** (Complete)
- ✅ MEDLINE XML parsing with XPath
- ✅ Publication Type → label mapping (10 canonical labels)
- ✅ MeSH term integration for Human and Animal labels
- ✅ Keyword-based backfill for InVitro/Human labels
- ✅ Label distribution analysis and validation
- ✅ Over-matching detection and correction (Animal label fixed)
- ✅ Multi-label combination analysis
- ✅ Data filtering and saving (64,981 labeled articles)
- ✅ Output: `data/processed/dental_abstracts.parquet`

**03 - EDA and Temporal Splits** (Next)
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
- ✅ NCBI API rate limiting and maintenance detection
- ✅ YAML configuration with multi-line string cleanup
- ✅ NCBI 10K result pagination limit (max 10,000 records per query)
- ✅ JSON decoding errors from control characters in queries
- ✅ Over-matching with broad keywords (Animal label: 81.2% → 7.9%)
- ✅ Balancing MeSH terms vs keyword matching for label accuracy
- ✅ Multi-label combination validation (Animal+Human, Human+InVitro)
- ✅ XML parsing with XPath for nested structures
- ✅ Label distribution analysis and quality validation
- 🔄 Handling class imbalance in medical literature
- 🔄 Preventing temporal leakage in train/test splits
- 🔄 Optimizing thresholds for multi-label predictions

---

## 📊 Progress Log

### 2024-11-08: Notebook 01 - PubMed Ingestion ✅ (Complete)

**Completed:**
- ✅ Environment setup with `python-dotenv` for NCBI credentials
- ✅ YAML configuration loading with `load_config()` function
- ✅ Query building with `build_query()` - handles multi-line YAML and year templating
- ✅ ESearch API integration with `esearch()` - fetches PMID lists from PubMed
- ✅ Pagination with `get_all_pmids()` - collects up to 10,000 PMIDs per year with progress tracking
- ✅ Error handling for NCBI maintenance windows and malformed queries
- ✅ Batch XML retrieval with `efetch()` function
- ✅ Complete ingestion pipeline for 2018-2025 (400 XML files)
- ✅ 76,165 articles successfully ingested

**Key Learnings:**
- NCBI E-utilities has a hard limit of 10,000 results per query
- Multi-line YAML strings need whitespace cleanup for API compatibility
- Environment variables require explicit loading with `python-dotenv`
- Rate limiting: 3 req/sec without API key, 10 req/sec with key
- Batch processing with 200 PMIDs per efetch call is efficient

**Dataset Size:**
- 76,165 articles total across 2018-2025
- 400 XML files stored in `data/raw/`
- Sufficient for multi-label classifier training

---

### 2024-11-08: Notebook 02 - Normalize and Label ✅ (Complete)

**Completed:**
- ✅ MEDLINE XML parsing with XPath expressions
- ✅ DataFrame creation with normalized structure (pmid, title, abstract, journal, year, pub_types, mesh_terms)
- ✅ Publication Type → label mapping with YAML configuration
- ✅ MeSH term integration for Human and Animal labels (more reliable than keywords)
- ✅ Keyword-based backfill for InVitro and Human labels
- ✅ Label assignment function with PT → MeSH → Keywords priority
- ✅ Label distribution analysis and validation
- ✅ Over-matching detection and correction (Animal: 81.2% → 7.9%)
- ✅ Multi-label combination analysis (Animal+Human, Human+InVitro validated)
- ✅ Data filtering (removed 14.7% unlabeled articles)
- ✅ Saved labeled dataset to `data/processed/dental_abstracts.parquet`

**Key Learnings:**
- Getting accurate labels requires careful data engineering, not just XML parsing
- MeSH terms are more reliable than keyword matching (manually curated by experts)
- Overly broad keywords (e.g., "in vivo", "animal study") can cause significant over-matching
- Iterative refinement is essential: identify problems → analyze → refine → validate
- Multi-label combinations like "Animal + Human" are legitimate (comparative/translational studies)
- 14.7% of articles don't fit study design categories (narrative reviews, editorials, etc.) - this is expected

**Dataset Quality:**
- **Total Articles:** 76,165
- **Labeled Articles:** 64,981 (85.3% coverage)
- **Label Distribution:** Realistic and aligned with evidence hierarchy
- **Animal Label:** Perfect match with MeSH (7.9% = 6,019 articles)
- **Human Label:** Good coverage (74.6% = 56,804 articles)
- **Multi-label:** 1.24 average labels per article, 37.6% have 2+ labels
- **Output File:** `data/processed/dental_abstracts.parquet` (ready for Notebook 03)

**Challenges Solved:**
- Fixed Animal label over-matching by removing overly broad keywords, relying solely on MeSH "Animals" term
- Refined Human label keywords to reduce false positives
- Validated multi-label combinations to ensure clinical/biological relevance
- Created reproducible labeling pipeline that can feed different Periospot AI apps

**Next Steps:**
- Notebook 03: EDA and temporal splits
- Analyze label distribution across years
- Create train/val/test splits (≤2021 / 2022-2023 / ≥2024)

---

## 📄 License

MIT License (see [LICENSE](LICENSE))

---

<div align="center">

**⭐ Star this repo if you found it helpful! ⭐**

*Building AI solutions for evidence-based dentistry, one abstract at a time* 🦷🤖

</div>

