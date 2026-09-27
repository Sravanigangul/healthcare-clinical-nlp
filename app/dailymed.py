"""Retrieve medication safety evidence from DailyMed SPL labels."""

from xml.etree import ElementTree as ET

import requests


DAILYMED_BASE_URL = "https://dailymed.nlm.nih.gov/dailymed/services/v2"


def search_labels(drug_name):
    """Search DailyMed for SPL labels matching a medication name."""

    response = requests.get(
        f"{DAILYMED_BASE_URL}/spls.json",
        params={"drug_name": drug_name},
        timeout=15,
    )

    response.raise_for_status()

    data = response.json()

    return data.get("data", [])


def find_single_ingredient_labels(drug_name, labels):
    """Prefer labels that appear to contain the requested drug alone."""

    drug_name = drug_name.lower()
    matches = []

    for label in labels:
        title = label.get("title", "")
        title_lower = title.lower()

        if not title_lower.startswith(drug_name):
            continue

        product_name = title_lower.split("[")[0]

        if " and " in product_name:
            continue

        matches.append(label)

    return matches


def get_label_xml(setid):
    """Retrieve a DailyMed SPL document by Set ID."""

    response = requests.get(
        f"{DAILYMED_BASE_URL}/spls/{setid}.xml",
        timeout=15,
    )

    response.raise_for_status()

    return response.text


def extract_safety_sections(xml_text):
    """Extract medication-safety sections from a DailyMed SPL document."""

    safety_codes = {
        "34073-7": "drug_interactions",
        "34071-1": "warnings",
        "50570-1": "do_not_use",
        "50569-3": "ask_doctor",
        "50568-5": "ask_doctor_or_pharmacist",
        "50566-9": "stop_use",
        "53414-9": "pregnancy_or_breastfeeding",
    }

    root = ET.fromstring(xml_text)
    sections = {}

    for section in root.iter():
        if not section.tag.endswith("section"):
            continue

        code_element = next(
            (
                child
                for child in section
                if child.tag.endswith("code")
            ),
            None,
        )

        if code_element is None:
            continue

        code = code_element.get("code")

        if code not in safety_codes:
            continue

        text = " ".join(
            part.strip()
            for part in section.itertext()
            if part.strip()
        )

        sections[safety_codes[code]] = text

    return sections


def build_safety_evidence(drug_name, label, safety_sections):
    """Build structured medication-safety evidence from a DailyMed label."""

    return {
        "drug": drug_name,
        "source": "DailyMed",
        "setid": label.get("setid"),
        "label_title": label.get("title"),
        "match_type": "reference_label",
        "published_date": label.get("published_date"),
        "evidence": safety_sections,
    }


def select_best_label(medication, labels):
    """Prefer a DailyMed label matching the extracted medication dose."""

    if not labels:
        return None

    dose = medication.get("dose")
    unit = medication.get("unit")

    if dose and unit:
        dose_text = f"{dose} {unit}".lower()

        dose_matches = [
            label
            for label in labels
            if dose_text in label.get("title", "").lower()
        ]

        if dose_matches:
            return dose_matches[0]

    return labels[0]


def enrich_medication_with_safety(medication):
    """Enrich one medication with DailyMed safety evidence."""

    drug_name = (
        medication.get("normalized_name")
        or medication.get("drug")
    )

    if not drug_name:
        return medication

    labels = search_labels(drug_name)

    single_labels = find_single_ingredient_labels(
        drug_name,
        labels,
    )

    if not single_labels:
        return {
            **medication,
            "safety_evidence": None,
        }

    label = select_best_label(
        medication,
        single_labels,
    )

    xml_text = get_label_xml(
        label["setid"]
    )

    safety_sections = extract_safety_sections(
        xml_text
    )

    safety_evidence = build_safety_evidence(
        drug_name,
        label,
        safety_sections,
    )

    return {
        **medication,
        "safety_evidence": safety_evidence,
    }


def enrich_medications_with_safety(medications):
    """Enrich a medication list with DailyMed safety evidence."""

    enriched_medications = []

    for medication in medications:
        enriched = enrich_medication_with_safety(
            medication
        )

        enriched_medications.append(
            enriched
        )

    return enriched_medications