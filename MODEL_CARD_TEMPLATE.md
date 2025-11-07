# Model Card — Dental Evidence Triage (DistilBERT, Multi-label)

## Model Details

### Basic Information

- **Model Name:** `dental-evidence-triage`
- **Model Version:** 1.0
- **Model Type:** Multi-label Text Classification
- **Base Architecture:** DistilBERT (distilbert-base-uncased)
- **Framework:** PyTorch + Hugging Face Transformers
- **Author:** Francisco Teixeira Barbosa (@Tuminha)
- **Date:** November 2025
- **License:** MIT

### Model Description

A fine-tuned DistilBERT model that classifies dental research abstracts into study-design categories. Given a **title + abstract**, the model predicts one or more of 10 canonical labels:

1. SystematicReview
2. MetaAnalysis
3. RCT (Randomized Controlled Trial)
4. ClinicalTrial
5. Cohort
6. CaseControl
7. CaseReport
8. InVitro
9. Animal
10. Human

**Why Multi-label?**
- Papers can have multiple study characteristics (e.g., RCT + Human)
- Systematic reviews may also be meta-analyses
- Some studies combine animal and in vitro work

---

## Intended Use

### Primary Use Cases

- **Literature Triage:** Quickly classify PubMed abstracts by evidence type
- **Systematic Review Screening:** Pre-filter abstracts before manual review
- **Research Databases:** Auto-tag papers for evidence hierarchies
- **Educational Tools:** Teach students to identify study designs

### Intended Users

- Researchers conducting systematic reviews
- Librarians and information specialists
- Clinical guideline developers
- Dental educators and students
- AI developers building knowledge pipelines

---

## Training Data

### Data Source

- **Dataset:** PubMed/MEDLINE dental abstracts (2018–2025)
- **Total Records:** ~50,000–100,000 (varies by query scope)
- **Training Split:** ≤2021 (~60-70%)
- **Validation Split:** 2022-2023 (~15-20%)
- **Test Split:** ≥2024 (~15-20%)

### Labeling Strategy

Labels derived from **MEDLINE Publication Types** (PT) with keyword backfill:
- **Silver Labels:** Not manually annotated; derived from structured metadata
- **Expected Noise:** ~5-10% due to indexing inconsistencies

See [DATACARD.md](DATACARD.md) for full documentation.

### Preprocessing

- **Input:** Concatenated `title + " " + abstract`
- **Tokenization:** DistilBERT tokenizer (max_length=512, truncation=True)
- **Label Encoding:** Multi-hot binary vectors (10 dimensions)

---

## Training Details

### Hyperparameters

- **Learning Rate:** 5e-5
- **Batch Size:** 8 (with gradient accumulation if needed)
- **Epochs:** 3–4
- **Optimizer:** AdamW
- **Loss Function:** BCEWithLogitsLoss (binary cross-entropy for multi-label)
- **Warmup Steps:** 10% of total training steps
- **Weight Decay:** 0.01

### Evaluation Strategy

- **Validation Frequency:** Every epoch
- **Early Stopping:** Based on micro-F1 on validation set
- **Best Model Selection:** Highest micro-F1

### Computational Resources

- **Hardware:** [Specify: e.g., 1x NVIDIA T4, 16GB RAM]
- **Training Time:** ~2-4 hours (depending on dataset size)

---

## Evaluation Metrics

### Aggregate Metrics (Test Set)

*Fill in after training:*

| Metric | Value |
|--------|-------|
| **Micro-F1** | [TARGET: ≥0.75] |
| **Macro-F1** | [Expected lower due to imbalance] |
| **Micro-Precision** | [TBD] |
| **Micro-Recall** | [TBD] |
| **Macro-Precision** | [TBD] |
| **Macro-Recall** | [TBD] |

### Per-label Performance

*Example template:*

| Label | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| SystematicReview | 0.XX | 0.XX | 0.XX | XXX |
| MetaAnalysis | 0.XX | 0.XX | 0.XX | XXX |
| RCT | 0.XX | 0.XX | 0.XX | XXX |
| ClinicalTrial | 0.XX | 0.XX | 0.XX | XXX |
| Cohort | 0.XX | 0.XX | 0.XX | XXX |
| CaseControl | 0.XX | 0.XX | 0.XX | XXX |
| CaseReport | 0.XX | 0.XX | 0.XX | XXX |
| InVitro | 0.XX | 0.XX | 0.XX | XXX |
| Animal | 0.XX | 0.XX | 0.XX | XXX |
| Human | 0.XX | 0.XX | 0.XX | XXX |

### Threshold Optimization

- **Default Threshold:** 0.5 (probability cutoff for positive prediction)
- **Optimized Thresholds:** [Per-label thresholds tuned on validation set, if applicable]

---

## Limitations

### Data Limitations

- **Silver Labels:** Derived from metadata, not expert annotation
- **Temporal Lag:** Newest papers may have incomplete Publication Types
- **Language:** Trained primarily on English abstracts
- **Missing Abstracts:** ~5-10% of PubMed records lack abstracts (excluded)

### Model Limitations

- **Class Imbalance:** Underperforms on rare labels (CaseControl, MetaAnalysis)
- **Ambiguity:** Difficulty distinguishing RCT from ClinicalTrial
- **Context:** Limited to title + abstract (no full-text analysis)
- **Domain:** Optimized for dental research; generalization to other medical fields untested

### Technical Limitations

- **Max Length:** Truncates abstracts >512 tokens
- **Vocabulary:** DistilBERT's 30K wordpiece tokens may miss rare dental terminology
- **Inference Speed:** ~50-100ms per abstract on CPU; faster on GPU

---

## Ethical Considerations

### Intended Assistive Use

- **Not Diagnostic:** Do not use as sole evidence for clinical decisions
- **Screening Tool:** Predictions should be validated by domain experts
- **Transparency:** Always disclose model-assisted screening in systematic reviews

### Potential Biases

- **Publication Bias:** Model inherits bias toward published (often positive) results
- **Language Bias:** English-language training data overrepresents Western research
- **Indexing Bias:** MEDLINE PT assignment varies by journal prestige and topic
- **Temporal Bias:** Training data ≤2021 may not capture emerging study designs

### Failure Modes

- **False Negatives (RCT):** May miss poorly described randomized trials
- **False Positives (SystematicReview):** May overpredict on literature review narratives
- **Label Confusion:** RCT vs ClinicalTrial, Cohort vs CaseControl

### Recommendations

- Use as **first-pass filter**, not replacement for expert review
- Validate predictions on a random sample for quality assurance
- Re-train annually to capture evolving research practices

---

## Out of Scope

This model **does NOT** support:

- **Quality Assessment:** No risk-of-bias or GRADE scoring
- **Causal Inference:** Cannot determine treatment efficacy
- **Non-English Abstracts:** Untested on non-English text
- **Full-Text Analysis:** Limited to abstracts
- **Real-time Diagnosis:** Not validated for clinical use

---

## Model Access & Inference

### Hugging Face Hub

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "Tuminha/dental-evidence-triage"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Example inference
text = "Title: Effect of dental implants on bone density. Abstract: This randomized controlled trial..."
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
outputs = model(**inputs)
probs = torch.sigmoid(outputs.logits)[0].tolist()

labels = ["SystematicReview", "MetaAnalysis", "RCT", "ClinicalTrial", "Cohort", 
          "CaseControl", "CaseReport", "InVitro", "Animal", "Human"]
predictions = {label: prob for label, prob in zip(labels, probs) if prob > 0.5}
print(predictions)
```

### Gradio Demo

See `notebooks/07_inference_demo.ipynb` for an interactive interface.

---

## Maintenance & Updates

- **Re-training Frequency:** Annually or when label distribution shifts
- **Data Refresh:** Quarterly PubMed ingestion for new papers
- **Model Versioning:** Track versions with date stamps (e.g., `v1.0_2025-11`)

---

## Citation

If you use this model, please cite:

```
Barbosa, F. T. (2025). Dental Evidence Triage: A Multi-label Study-Design 
Classifier for PubMed Dental Abstracts. Hugging Face Model Hub: 
https://huggingface.co/Tuminha/dental-evidence-triage
```

---

## Contact

**Francisco Teixeira Barbosa**
- Email: cisco@periospot.com
- GitHub: [@Tuminha](https://github.com/Tuminha)
- Twitter: [@cisco_research](https://twitter.com/cisco_research)

For questions, issues, or collaboration inquiries, please open an issue on the [GitHub repository](https://github.com/Tuminha/dental-evidence-triage).

---

## Acknowledgments

- **Data Source:** NCBI PubMed/MEDLINE
- **Base Model:** Hugging Face DistilBERT
- **Inspiration:** Evidence-based dentistry and systematic review methodology

---

## Changelog

### v1.0 (November 2025)
- Initial release
- Training on 2018-2025 dental abstracts
- 10-label multi-label classification
- Temporal train/val/test splits

