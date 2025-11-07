# Data Card — Dental Evidence Triage

## Dataset Description

### Source

**PubMed E-utilities (NCBI)**, 2018–2025 (configurable in `configs/query.yaml`)

- **Database:** PubMed/MEDLINE
- **Access Method:** NCBI E-utilities API (ESearch + EFetch)
- **Query Scope:** Dental research literature
- **Languages:** English (by default; configurable)

### Fields

Each record contains:

| Field | Type | Description |
|-------|------|-------------|
| `pmid` | string | PubMed unique identifier |
| `title` | string | Article title |
| `abstract` | string | Abstract text (may be empty) |
| `journal` | string | Journal name |
| `year` | integer | Publication year |
| `pub_types` | list[string] | Publication Types from MEDLINE |
| `mesh_terms` | list[string] | MeSH (Medical Subject Headings) |
| `labels` | list[string] | Canonical study-design labels (derived) |
| `split` | string | train / val / test |

### Query Specification

Default query template (see `configs/query.yaml`):

```
(
  dentistry[MeSH Terms] OR dental[Title/Abstract] OR "oral surgery"[Title/Abstract]
  OR periodont*[Title/Abstract] OR implant*[Title/Abstract]
  OR prosthodont*[Title/Abstract] OR endodont*[Title/Abstract]
  OR orthodont*[Title/Abstract] OR "maxillofacial"[Title/Abstract]
) AND (YYYY[PDAT]:YYYY[PDAT])
```

This targets:
- Periodontology
- Dental implants
- Prosthodontics
- Endodontics
- Orthodontics
- Oral and maxillofacial surgery
- General dentistry

---

## Labeling Strategy

### Canonical Labels (10 categories)

Multi-label classification with the following targets:

1. **SystematicReview** — Systematic reviews
2. **MetaAnalysis** — Meta-analyses
3. **RCT** — Randomized Controlled Trials
4. **ClinicalTrial** — Non-randomized clinical trials
5. **Cohort** — Cohort studies (prospective/retrospective)
6. **CaseControl** — Case-control studies
7. **CaseReport** — Case reports
8. **InVitro** — In vitro or ex vivo studies
9. **Animal** — Animal studies
10. **Human** — Human subjects (not mutually exclusive with other labels)

### Mapping Logic

Labels are derived from **Publication Types** (PT) assigned by MEDLINE indexers, with keyword-based backfill for gaps:

```yaml
# Example from configs/pt_to_labels.yaml
SystematicReview:
  pt: ["Systematic Review"]

InVitro:
  pt: []
  keywords: ["in vitro", "ex vivo"]

Animal:
  pt: ["Animal Experimentation"]
  keywords: ["rat", "mouse", "murine", "canine", "rabbit", "dog", "porcine", "animal study"]
```

**Why Multi-label?**
- A paper can be both RCT and Human
- Systematic reviews may include Meta-Analysis
- Some studies combine Animal and InVitro work

### Label Quality

- **Silver Labels:** Derived from structured metadata (not manual annotation)
- **Expected Noise:** ~5-10% due to:
  - Inconsistent MEDLINE indexing
  - Lag in PT assignment for new papers
  - Ambiguity between ClinicalTrial and RCT
  - Keyword-based rules for InVitro/Animal

---

## Data Splits

### Temporal Splitting Strategy

To prevent **temporal leakage** (where future knowledge influences past predictions):

| Split | Years | Purpose |
|-------|-------|---------|
| **Train** | ≤ 2021 | Model training |
| **Validation** | 2022–2023 | Hyperparameter tuning, threshold selection |
| **Test** | ≥ 2024 | Final evaluation (held-out) |

**Rationale:**
- Research trends evolve over time
- Prevents overfitting to contemporary phrasing
- Mimics real-world deployment (predicting future papers)

### Expected Distribution

Approximate class balance (varies by query scope):

- **Common:** CaseReport (30-40%), Human (50-60%)
- **Moderate:** RCT (5-10%), Cohort (10-15%), InVitro (10-15%)
- **Rare:** MetaAnalysis (2-5%), SystematicReview (3-5%)
- **Very Rare:** CaseControl (1-3%)

---

## Known Issues & Limitations

### Missing Data

- **Abstracts:** ~5-10% of records lack abstracts (excluded during preprocessing)
- **Unlabeled Papers:** ~10-20% of recent papers may not have Publication Types assigned yet

### Class Imbalance

- Severe imbalance favoring CaseReport and Human
- Few examples of CaseControl and MetaAnalysis
- **Mitigation:** Threshold optimization, class weighting, or focal loss

### Temporal Lag

- Newest papers (2024-2025) may have incomplete metadata
- MEDLINE indexing can lag by several months

### Ambiguity

- **RCT vs ClinicalTrial:** Overlap in MEDLINE indexing
- **Cohort vs Observational:** Some retrospective studies ambiguously labeled
- **InVitro + Animal:** Studies may involve both

### Language Bias

- Default query excludes non-English papers
- May underrepresent global dental research

---

## Intended Use

### Primary Uses

- **Literature Triage:** Quickly filter PubMed results by study design
- **Systematic Review Screening:** Pre-filter abstracts before full-text review
- **Curriculum Datasets:** Generate study-design balanced teaching sets
- **Retrieval Enhancement:** Augment PubMed searches with predicted study types

### Secondary Uses

- **Trend Analysis:** Track evolution of study designs over time
- **Research Gaps:** Identify underrepresented study types in subdomains
- **Training Data:** Seed for more sophisticated evidence-grading models

---

## Out of Scope

This dataset and model **do NOT** support:

- **Quality Assessment:** No risk-of-bias scoring
- **Causal Claims:** Cannot determine treatment efficacy
- **Full-Text Analysis:** Limited to title + abstract
- **Diagnostic Use:** Not validated for clinical decision-making
- **Non-English Literature:** Default scope is English-only

---

## Ethical Considerations

### Bias Sources

- **Publication Bias:** Positive results more likely to be published
- **Language Bias:** English-language preference
- **Geographic Bias:** PubMed overrepresents Western research
- **Indexing Bias:** MEDLINE PT assignment varies by journal/topic

### Responsible Use

- **Assistive, Not Authoritative:** Treat predictions as screening aids, not ground truth
- **Human Verification:** Always validate model outputs with domain expertise
- **Transparency:** Disclose model limitations when using in systematic reviews
- **No Patient Data:** This dataset contains only published research metadata

---

## Maintenance & Updates

- **Quarterly Refresh:** Re-run ingestion pipeline to capture new papers
- **Label Schema Stability:** Avoid frequent changes to canonical labels
- **Versioning:** Track dataset versions with date stamps (e.g., `dental_abstracts_2025_11.parquet`)

---

## Citation

If you use this dataset, please cite:

```
Barbosa, F. T. (2025). Dental Evidence Triage: A Multi-label Study-Design 
Classifier for PubMed Dental Abstracts. GitHub repository: 
https://github.com/Tuminha/dental-evidence-triage
```

And acknowledge the original data source:

```
NCBI Resource Coordinators. (2016). Database resources of the National Center 
for Biotechnology Information. Nucleic Acids Research, 44(D1), D7-D19.
```

---

## Contact

**Francisco Teixeira Barbosa**
- Email: cisco@periospot.com
- GitHub: [@Tuminha](https://github.com/Tuminha)
- Twitter: [@cisco_research](https://twitter.com/cisco_research)

For questions about data provenance, labeling logic, or dataset access, please open an issue in the GitHub repository.

