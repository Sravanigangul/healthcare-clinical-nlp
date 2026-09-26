"""Utilities for loading and linking clinical concepts to ICD-10-CM terminology."""
import torch
from transformers import AutoTokenizer, AutoModel


def load_icd10_codes(file_path):
    """Load valid ICD-10-CM codes and descriptions from the official code file."""

    records = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            code = line[:7].strip()
            description = line[8:].strip()

            if code and description:
                records.append({
                    "code": code,
                    "description": description,
                })

    return records

def load_sapbert_model(
    model_name="cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
):
    """Load the SapBERT tokenizer and model for biomedical concept linking."""

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    model.eval()

    return tokenizer, model

def encode_concepts(
    texts,
    tokenizer,
    model,
):
    """Convert biomedical concepts into SapBERT embeddings."""

    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(**inputs)

    embeddings = outputs.last_hidden_state[:, 0, :]

    embeddings = torch.nn.functional.normalize(
        embeddings,
        p=2,
        dim=1,
    )

    return embeddings

def encode_icd_descriptions(
    records,
    tokenizer,
    model,
    batch_size=64,
):
    """Encode ICD-10-CM descriptions in batches."""

    batches = []

    for start in range(0, len(records), batch_size):
        batch_records = records[start:start + batch_size]

        descriptions = [
            record["description"]
            for record in batch_records
        ]

        embeddings = encode_concepts(
            descriptions,
            tokenizer,
            model,
        )

        batches.append(embeddings)

        processed = min(start + batch_size, len(records))
        print(f"Encoded {processed}/{len(records)} ICD descriptions")

    return torch.cat(batches, dim=0)

def save_icd_embeddings(embeddings, file_path):
    """Save precomputed ICD-10-CM embeddings to disk."""

    torch.save(embeddings, file_path)

def load_icd_embeddings(file_path):
    """Load precomputed ICD-10-CM embeddings from disk."""

    return torch.load(
        file_path,
        map_location="cpu",
    )

def find_icd_candidates(
    query,
    records,
    icd_embeddings,
    tokenizer,
    model,
    top_k=5,
):
    """Find the most semantically similar ICD-10-CM candidates."""

    query_embedding = encode_concepts(
        [query],
        tokenizer,
        model,
    )

    scores = torch.matmul(
        icd_embeddings,
        query_embedding[0],
    )

    top_indices = torch.topk(
        scores,
        k=top_k,
    ).indices.tolist()

    results = []

    for index in top_indices:
        results.append({
            "code": records[index]["code"],
            "description": records[index]["description"],
            "similarity": scores[index].item(),
        })

    return results

def add_icd_candidates(
    patient_record,
    icd_records,
    icd_embeddings,
    tokenizer,
    model,
    top_k=5,
):
    """Add ICD-10-CM candidate suggestions for extracted conditions."""

    candidates = []

    for condition in patient_record.get("conditions", []):
        if not condition:
            continue

        matches = find_icd_candidates(
            query=condition,
            records=icd_records,
            icd_embeddings=icd_embeddings,
            tokenizer=tokenizer,
            model=model,
            top_k=top_k,
        )

        candidates.append({
            "condition": condition,
            "candidates": matches,
        })

    patient_record["icd10_candidates"] = candidates

    return patient_record