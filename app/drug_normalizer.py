"""Utilities for normalizing medication names using the RxNorm API."""

import requests

from app.dailymed import enrich_medications_with_safety


RXNORM_BASE_URL = "https://rxnav.nlm.nih.gov/REST"


def normalize_drug(drug_name):
    """Normalize a medication name using RxNorm."""

    response = requests.get(
        f"{RXNORM_BASE_URL}/rxcui.json",
        params={
            "name": drug_name,
            "search": 2,
        },
        timeout=10,
    )

    response.raise_for_status()

    data = response.json()

    rxnorm_ids = data.get(
        "idGroup",
        {},
    ).get(
        "rxnormId",
        [],
    )

    if not rxnorm_ids:
        return {
            "drug": drug_name,
            "rxcui": None,
            "normalized_name": None,
            "term_type": None,
        }

    rxcui = rxnorm_ids[0]

    properties_response = requests.get(
        f"{RXNORM_BASE_URL}/rxcui/{rxcui}/properties.json",
        timeout=10,
    )

    properties_response.raise_for_status()

    properties = properties_response.json().get(
        "properties",
        {},
    )

    return {
        "drug": drug_name,
        "rxcui": rxcui,
        "normalized_name": properties.get("name"),
        "term_type": properties.get("tty"),
    }


def normalize_medications(medication_list):
    """Normalize extracted medications using RxNorm."""

    normalized_medications = []

    for medication in medication_list:
        drug_name = medication.get("drug")

        if not drug_name:
            continue

        normalized = normalize_drug(drug_name)

        normalized_medications.append({
            **medication,
            "rxcui": normalized["rxcui"],
            "normalized_name": normalized["normalized_name"],
            "term_type": normalized["term_type"],
        })

    return normalized_medications


def normalize_patient_medications(patient_record):
    """Enrich patient-record medications with RxNorm identifiers."""

    medications = patient_record.get("medications", [])

    patient_record["medications"] = normalize_medications(
        medications
    )

    return patient_record


def enrich_patient_medications(patient_record):
    """Add RxNorm normalization and DailyMed safety evidence."""

    normalized_record = normalize_patient_medications(
        patient_record
    )

    medications = normalized_record.get(
        "medications",
        [],
    )

    normalized_record["medications"] = enrich_medications_with_safety(
        medications
    )

    return normalized_record