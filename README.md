# Healthcare Clinical NLP

A hands-on portfolio exploring Natural Language Processing (NLP) techniques for clinical and healthcare text.

This repository documents my progression from foundational clinical text processing and rule-based NLP to text classification, word embeddings, transformer-based representations, ICD-10 code retrieval, and medication extraction.

## Project Overview

Clinical data contains a large amount of valuable information in unstructured text such as clinical notes, discharge summaries, diagnoses, and medication documentation.

This project explores different approaches for transforming that unstructured text into structured and machine-readable information that can support downstream healthcare analytics and machine learning tasks.

The repository progresses from traditional NLP approaches to modern embedding and transformer-based methods.

## Clinical NLP Workflow

Clinical Text  
↓  
Text Cleaning & Normalization  
↓  
Clinical Entity / Information Extraction  
↓  
Feature Representation  
↓  
Text Classification & Semantic Similarity  
↓  
Transformer Embeddings  
↓  
Clinical Applications

## Notebooks

### 1. Clinical Text Processing

**Notebook:** `Clinical_Text_Processing.ipynb`

Foundational preprocessing techniques for clinical text, including:

- Text cleaning and normalization
- Clinical abbreviation handling
- Regular expressions
- Clinical section extraction
- Structured information extraction
- Preparation of clinical text for downstream NLP tasks

### 2. Clinical Named Entity Recognition

**Notebook:** `Clinical_Named_Entity_Recognition.ipynb`

Explores identification of clinically meaningful entities from unstructured text.

Topics include:

- Clinical entity extraction
- Rule-based pattern matching
- Regular-expression-based extraction
- Clinical terminology recognition
- Negation handling
- Evaluation of extracted entities

### 3. Clinical Text Classification

**Notebook:** `Clinical Text Classification.ipynb`

Explores traditional machine learning approaches for classifying clinical text.

Topics include:

- Bag-of-Words representations
- TF-IDF
- Unigrams and n-grams
- Logistic Regression
- Naive Bayes
- Linear SVM
- Model comparison
- ROC-AUC and F1 evaluation
- Clinical note classification
- Readmission-related text modeling

This notebook demonstrates how unstructured clinical text can be transformed into numerical features and used with supervised machine learning models.

### 4. Word Embeddings & Word2Vec

**Notebook:** `Word Embeddings & Word2Vec.ipynb`

Explores distributed word representations and semantic relationships between clinical terms.

Topics include:

- Word2Vec
- Skip-gram
- Continuous Bag of Words (CBOW)
- Context windows
- Vocabulary construction
- Cosine similarity
- Clinical word relationships
- Word analogy experiments
- Document vectors using averaged word embeddings
- Dimensionality reduction
- PCA and t-SNE visualization

The notebook demonstrates how embeddings capture relationships between clinical terms beyond simple word-frequency representations.

### 5. ClinicalBERT & Transformer Embeddings

**Notebook:** `ClinicalBERT.ipynb`

Introduces transformer-based NLP and contextual embeddings for clinical text.

Topics include:

- BERT architecture
- Bidirectional self-attention
- WordPiece tokenization
- Contextual embeddings
- SentenceTransformer embeddings
- Cosine similarity between clinical notes
- PCA visualization of embedding spaces
- Zero-shot classification
- ClinicalBERT
- BioBERT and PubMedBERT
- ClinicalBERT fine-tuning workflow
- Readmission classification concepts

This notebook explores the transition from static word embeddings such as Word2Vec to contextual transformer-based representations.

### 6. ICD Coding & Medication Extraction

**Notebook:** `ICD Coding & Medication Extraction.ipynb`

Applies NLP techniques to clinical coding and medication-related tasks.

Topics include:

- TF-IDF-based ICD-10 candidate retrieval
- Cosine-similarity ranking
- ICD-10 format validation
- Structured medication extraction
- Drug name recognition
- Dose extraction
- Medication units
- Route and frequency extraction
- Medication reconciliation
- Identification of continued medications
- Identification of discontinued medications
- New medications at discharge
- Dose-change detection
- Rule-based drug-drug interaction checking

This notebook demonstrates how NLP can be combined with clinical reference data and rule-based logic to create structured healthcare information from free text.

## Methods Explored

The repository covers several generations of NLP techniques:

### Rule-Based NLP
- Regular expressions
- Clinical dictionaries
- Pattern matching
- Negation detection
- Medication extraction

### Traditional NLP & Machine Learning
- Bag of Words
- TF-IDF
- N-grams
- Logistic Regression
- Naive Bayes
- Linear SVM

### Word Embeddings
- Word2Vec
- Skip-gram
- CBOW
- Cosine similarity
- Document embeddings

### Transformer-Based NLP
- BERT concepts
- Sentence Transformers
- ClinicalBERT
- Contextual embeddings
- Zero-shot classification
- Fine-tuning concepts

## Technologies

- Python
- Pandas
- NumPy
- Regular Expressions (Regex)
- Scikit-learn
- Matplotlib
- spaCy
- Gensim
- Word2Vec
- Hugging Face Transformers
- Sentence Transformers
- PyTorch
- ClinicalBERT

## Key Concepts Practiced

Through these notebooks, I am developing practical understanding of:

- Processing unstructured clinical text
- Representing text numerically
- Extracting clinically meaningful information
- Building and evaluating text classifiers
- Understanding semantic similarity
- Working with static and contextual embeddings
- Applying transformer models to healthcare text
- Retrieving ICD-10 candidates from clinical descriptions
- Structuring medication information
- Connecting NLP techniques with healthcare use cases

## Repository Structure

    healthcare-clinical-nlp/
    │
    ├── notebooks/
    │   ├── Clinical_Text_Processing.ipynb
    │   ├── Clinical_Named_Entity_Recognition.ipynb
    │   ├── Clinical Text Classification.ipynb
    │   ├── Word Embeddings & Word2Vec.ipynb
    │   ├── ClinicalBERT.ipynb
    │   └── ICD Coding & Medication Extraction.ipynb
    │
    └── README.md

## Project Status

This repository is an ongoing learning and portfolio project. Additional experiments will focus on applying these NLP techniques to larger public or de-identified healthcare datasets and developing more complete end-to-end clinical NLP workflows.

## Important Note

The examples in this repository are intended for educational and portfolio purposes. Some exercises use synthetic clinical text, manually created reference data, or simplified rule-based approaches.

The outputs should not be interpreted as clinical recommendations or used for patient care, medical coding, medication safety decisions, or other clinical decision-making without appropriate validation.

## Future Work

Planned areas for further development include:

- End-to-end clinical NLP pipelines
- Fine-tuning transformer models on labeled clinical data
- Long-document clinical text processing
- Improved clinical entity linking
- Semantic search and retrieval
- Evaluation on larger public or de-identified datasets
- Retrieval-Augmented Generation (RAG) for healthcare information
- Model evaluation, explainability, and error analysis

## Author

**Sravani Gangula**

Healthcare Data Analytics | Clinical NLP | Data Science | Healthcare AI
