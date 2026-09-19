# Healthcare Clinical NLP

An end-to-end Clinical NLP and Healthcare AI portfolio project for extracting structured clinical information from unstructured healthcare documents.

The project progresses from rule-based NLP and traditional machine learning to transformer-based clinical Named Entity Recognition (NER) using BioClinicalBERT, long-document processing, medication extraction, negation detection, and structured clinical document analysis.

The current system is being developed into a **Clinical Document Review Assistant** capable of processing clinical documents and organizing extracted information for downstream review.

---

## Project Overview

Clinical notes, discharge summaries, reports, and other healthcare documents contain valuable information in unstructured text.

This project explores how Natural Language Processing and machine learning can transform this text into structured information such as:

- Conditions and diagnoses
- Signs and symptoms
- Medications
- Laboratory values
- Diagnostic procedures
- Therapeutic procedures
- Demographics
- Negated clinical findings
- ICD-10 candidate codes
- Drug-drug interaction information

The project combines:

- Rule-based clinical NLP
- Traditional machine learning
- Word embeddings
- Transformer models
- BioClinicalBERT fine-tuning
- Long-document NER
- Structured clinical information extraction

---

## Clinical Document Intelligence Pipeline

```text
Clinical Document / PDF
          │
          ▼
    Text Extraction
          │
          ▼
   Clinical Preprocessing
          │
          ▼
 ┌───────────────────────┐
 │   BioClinicalBERT NER │
 └───────────┬───────────┘
             │
     Long-document inference
     512-token windows
     128-token overlap
             │
             ▼
      Clinical Entities
             │
    ┌────────┼─────────┐
    ▼        ▼         ▼
Conditions Symptoms Procedures
             │
             ▼
     Negation Detection
             │
             ▼
    Structured Patient Record
             │
    ┌────────┼───────────┐
    ▼        ▼           ▼
Medication  Labs      DDI Checking
Extraction
             │
             ▼
     Clinical Review Output
```

Future application layers will include ICD-10 candidate linking, expanded medication normalization and interaction retrieval, Retrieval-Augmented Generation (RAG), and grounded document summarization.

---

# BioClinicalBERT Clinical NER

A major component of this project is a supervised clinical NER model built by fine-tuning:

**BioClinicalBERT — `emilyalsentzer/Bio_ClinicalBERT`**

The model was trained using the MACCROBAT clinical NLP dataset.

## Dataset

- 200 clinical documents
- 140 training documents
- 30 validation documents
- 30 held-out test documents
- 41 clinical entity categories
- 82 observed BIO labels

The split was performed at the **document level** to prevent information leakage between training, validation, and test sets.

---

## Long-Document Processing

Many clinical documents exceed BERT's 512-token context limit.

Instead of truncating these documents, the pipeline uses:

- Maximum sequence length: 512 tokens
- Overlap stride: 128 tokens
- WordPiece-to-original-word alignment
- First-WordPiece label supervision
- Overlap-aware logit aggregation
- Document-level reconstruction

Training documents produced 286 model chunks and validation documents produced 64 chunks.

During document-level inference, predictions for words appearing in overlapping windows are aggregated before reconstructing the complete BIO sequence.

---

## Model Training

Selected configuration:

| Parameter | Value |
|---|---|
| Base model | BioClinicalBERT |
| Learning rate | 5e-5 |
| Epochs | 5 |
| Training batch size | 2 |
| Gradient accumulation | 2 |
| Effective batch size | 4 |
| Weight decay | 0.01 |
| Maximum sequence length | 512 |
| Stride | 128 |
| Mixed precision | FP16 |
| Random seed | 42 |

The best model was selected using validation entity-level F1 rather than test-set performance.

---

## Validation Performance

Using overlap-aware document-level reconstruction:

| Metric | Score |
|---|---:|
| Precision | 0.5505 |
| Recall | 0.6176 |
| F1 | 0.5821 |

---

## Held-Out Test Performance

The final selected model was evaluated once on the held-out test set after model selection.

| Metric | Score |
|---|---:|
| Precision | **0.5805** |
| Recall | **0.6500** |
| Entity-level Micro F1 | **0.6133** |

**Final held-out test F1: 61.33%**

This is an entity-level micro F1 score and should not be interpreted as classification accuracy.

---

## Model Error Analysis

Error analysis showed that the largest source of model error was not BIO boundary detection.

Among incorrect token-level predictions:

| Error Type | Percentage |
|---|---:|
| Wrong entity type | 49.32% |
| Entity ↔ non-entity | 41.19% |
| BIO boundary only | 9.49% |

Common semantic confusions included:

- Disease/disorder ↔ sign/symptom
- Diagnostic procedure ↔ detailed description
- Detailed description ↔ biological structure
- Diagnostic procedure ↔ biological structure
- Family history ↔ history

These results suggest that semantic differentiation between clinically related entity categories is a larger challenge than BIO boundary detection alone.

---

## Controlled Experiments

Several training configurations were evaluated using the validation set.

Experiments included:

- 3-epoch baseline
- 5-epoch training
- Learning rate comparison
- Class-weighted loss
- Per-entity evaluation
- Error analysis
- Overlap-aware document reconstruction

A 5-epoch model using a learning rate of `5e-5` provided the strongest validation performance among the evaluated configurations.

Class weighting increased recall slightly but reduced overall F1, so the unweighted model was retained.

---

# Clinical NLP Application

The application layer converts model output into a structured clinical record.

Current components include:

### BioClinicalBERT NER

Extracts clinical entity categories from text using the fine-tuned transformer model.

### Long-Document Inference

Processes documents using overlapping 512-token windows and aggregates predictions across overlapping regions.

### Medication Extraction

Extracts:

- Medication name
- Dose
- Unit
- Route
- Frequency

### Laboratory Extraction

Extracts selected laboratory measurements and their values/units.

### Negation Detection

Identifies findings preceded by common clinical negation expressions.

Example:

```text
Patient denies fever.
```

The extracted fever entity can be separated from active clinical findings.

### Drug-Drug Interaction Prototype

Extracted medications can currently be compared against a small educational interaction reference.

This component is intentionally limited and is not a comprehensive medication safety system.

### Structured Patient Record

The pipeline organizes extracted information into a structure similar to:

```json
{
  "demographics": {},
  "conditions": [],
  "symptoms": [],
  "medications": [],
  "labs": [],
  "diagnostic_procedures": [],
  "therapeutic_procedures": [],
  "negated_findings": [],
  "icd10_candidates": [],
  "drug_interactions": [],
  "entities": [],
  "summary": ""
}
```

---

# Notebooks

## 1. Clinical Text Processing

`Clinical_Text_Processing.ipynb`

Covers:

- Text cleaning
- Normalization
- Clinical abbreviations
- Regular expressions
- Section extraction
- Structured information extraction

---

## 2. Clinical Named Entity Recognition

`Clinical_Named_Entity_Recognition.ipynb`

Covers:

- Clinical entity extraction
- Pattern matching
- Rule-based NER
- Negation
- Evaluation

---

## 3. MACCROBAT Data Exploration

`MACCROBAT_Data_Exploration.ipynb`

Explores the labeled clinical corpus used for supervised NER development, including entity distributions and document-level dataset preparation.

---

## 4. Clinical NER Training

`Clinical_NER_Training.ipynb`

Contains the end-to-end BioClinicalBERT NER experiment:

- Dataset preparation
- BIO label construction
- WordPiece alignment
- Long-document chunking
- BioClinicalBERT fine-tuning
- Validation
- Controlled experiments
- Error analysis
- Document-level reconstruction
- Held-out test evaluation

---

## 5. Clinical Text Classification

`Clinical Text Classification.ipynb`

Explores:

- Bag of Words
- TF-IDF
- N-grams
- Logistic Regression
- Naive Bayes
- Linear SVM
- F1 and ROC-AUC evaluation

---

## 6. Word Embeddings & Word2Vec

`Word Embeddings & Word2Vec.ipynb`

Explores:

- Word2Vec
- Skip-gram
- CBOW
- Cosine similarity
- Document embeddings
- PCA
- t-SNE

---

## 7. ClinicalBERT

`ClinicalBERT.ipynb`

Explores:

- BERT architecture
- WordPiece tokenization
- Contextual embeddings
- ClinicalBERT
- BioBERT
- PubMedBERT
- Sentence Transformers
- Zero-shot classification
- Fine-tuning concepts

---

## 8. ICD Coding & Medication Extraction

`ICD Coding & Medication Extraction.ipynb`

Explores:

- ICD-10 candidate retrieval
- TF-IDF similarity
- Medication extraction
- Dose / unit / route / frequency
- Medication reconciliation
- Dose-change detection
- Prototype drug-drug interaction checking

---

# Technologies

### NLP & Machine Learning

- Python
- spaCy
- Scikit-learn
- Hugging Face Transformers
- PyTorch
- BioClinicalBERT
- Sentence Transformers
- Gensim / Word2Vec

### Data & Analysis

- Pandas
- NumPy
- Matplotlib

### NLP Methods

- Regex and rule-based NLP
- TF-IDF
- N-grams
- Word2Vec
- Contextual embeddings
- BIO tagging
- Transformer fine-tuning
- Long-document inference
- Semantic similarity
- Clinical NER

---

# Repository Structure

```text
healthcare-clinical-nlp/
│
├── app/
│   ├── app.py
│   ├── clinical_nlp.py
│   └── references.py
│
├── notebooks/
│   ├── Clinical_Text_Processing.ipynb
│   ├── Clinical_Named_Entity_Recognition.ipynb
│   ├── MACCROBAT_Data_Exploration.ipynb
│   ├── Clinical_NER_Training.ipynb
│   ├── Clinical Text Classification.ipynb
│   ├── Word Embeddings & Word2Vec.ipynb
│   ├── ClinicalBERT.ipynb
│   └── ICD Coding & Medication Extraction.ipynb
│
├── tests/
│
├── models/
│
├── requirements.txt
├── Dockerfile
└── README.md
```

---

# Current Development

The current development phase is focused on converting the NLP experiments into an end-to-end **Clinical Document Review Assistant**.

Planned pipeline:

```text
PDF Upload
   ↓
Page-Aware Text Extraction
   ↓
BioClinicalBERT NER
   ↓
Medication + Lab Extraction
   ↓
Negation Detection
   ↓
Drug Normalization
   ↓
ICD-10 Candidate Linking
   ↓
Drug Interaction Retrieval
   ↓
RAG Knowledge Retrieval
   ↓
Grounded LLM Summarization
   ↓
Clinical Review Dashboard
```

---

# Limitations

This project is an educational and portfolio system and is **not a clinical decision-support system**.

Important limitations include:

- The NER model was trained on a relatively small public clinical NLP dataset.
- Performance varies substantially across entity categories.
- Rare entity classes have limited training examples.
- Medication and DDI reference components are currently limited.
- ICD-10 candidate suggestions require further development and validation.
- Model-generated confidence does not represent calibrated clinical certainty.
- Clinical outputs require human review.

---

# Reproducibility Note

The original final trained checkpoint was stored in temporary Google Colab runtime storage and was lost when the runtime reset.

The preprocessing pipeline, selected hyperparameters, validation results, error analysis, and held-out test metrics were preserved.

The selected model can therefore be reproduced using the documented configuration and random seed, although GPU training can produce small numerical differences between runs.

The reported held-out test result corresponds to the original completed experiment.

---

# Future Work

- PDF and long-document ingestion
- ICD-10 entity linking
- RxNorm-style medication normalization
- Expanded drug-interaction retrieval
- Clinical relation extraction
- Evidence and page-level provenance
- Retrieval-Augmented Generation (RAG)
- Grounded clinical summarization
- Model explainability
- Application deployment

---

# Important Note

All examples and outputs are intended for educational, research, and portfolio purposes.

The system should not be used for patient care, diagnosis, medical coding, medication safety decisions, or other clinical decision-making without appropriate validation and professional review.

---

# Author

**Sravani Gangula**

Healthcare Data Science | Clinical NLP | Machine Learning | Healthcare AI