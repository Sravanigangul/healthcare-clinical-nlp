# Setup

This repository uses Python and Jupyter notebooks.

## 1. Create a virtual environment

### Conda

```bash
conda create -n clinical-nlp python=3.11 -y
conda activate clinical-nlp
```

### Or with `venv`

```bash
python -m venv .venv
```

Activate it, then continue below.

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 3. Install the spaCy English model

Some notebooks use spaCy. Install the small English pipeline with:

```bash
python -m spacy download en_core_web_sm
```

## 4. Start Jupyter

```bash
jupyter lab
```

## 5. Transformer model downloads

The first run of notebooks that use Hugging Face or Sentence Transformers may
download pretrained model files. Internet access is therefore required on the
first run unless the models are already cached locally.

A Hugging Face token is optional for public models, but authenticated requests
can provide higher rate limits.

## Notes

- Some examples in this repository use synthetic or manually created clinical text.
- Do not commit API keys, Hugging Face tokens, patient identifiers, PHI, or private clinical data.
- Large model weights and local datasets are excluded by `.gitignore`.
