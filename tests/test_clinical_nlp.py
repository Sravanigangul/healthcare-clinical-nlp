from app.clinical_nlp import (
    extract_meds,
    check_ddi,
    analyze_note,
    extract_clinical_sections,
    extract_section_items,
    link_section_concepts_to_values
)


def test_extract_single_medication():
    text = "metoprolol 50mg PO BID"

    meds = extract_meds(text)

    assert len(meds) == 1
    assert meds[0]["drug"] == "metoprolol"
    assert meds[0]["dose"] == "50"
    assert meds[0]["unit"] == "mg"
    assert meds[0]["route"] == "PO"
    assert meds[0]["frequency"] == "BID"


def test_extract_two_medications():
    text = "metoprolol 50mg PO BID and lisinopril 10mg PO daily"

    meds = extract_meds(text)

    assert len(meds) == 2

    drugs = [med["drug"] for med in meds]

    assert "metoprolol" in drugs
    assert "lisinopril" in drugs


def test_extract_warfarin_and_aspirin():
    text = "warfarin 5mg PO QD, aspirin 81mg PO QD"

    meds = extract_meds(text)

    drugs = [med["drug"] for med in meds]

    assert "warfarin" in drugs
    assert "aspirin" in drugs
    assert len(meds) == 2


def test_ddi_detection():
    medication_list = [
        {"drug": "warfarin"},
        {"drug": "aspirin"}
    ]

    interactions = check_ddi(medication_list)

    assert len(interactions) >= 1
    assert interactions[0]["drug1"] == "warfarin"
    assert interactions[0]["drug2"] == "aspirin"
    assert interactions[0]["severity"] == "HIGH"


def test_empty_note():
    result = analyze_note("")

    assert result["medications"] == []
    assert result["drug_interactions"] == []


def test_analyze_note_end_to_end():
    text = "warfarin 5 mg PO QD, aspirin 81mg PO QD"

    result = analyze_note(text)

    assert len(result["medications"]) == 2
    assert len(result["drug_interactions"]) >= 1

def test_section_extraction_and_linking():
    """Test section-aware extraction without loading the NER model."""

    text = """
PAST MEDICAL HISTORY
Hypertension
Type 2 diabetes mellitus

VITAL SIGNS
Blood pressure: 148/92 mmHg
Heart rate: 96 bpm

LABORATORY RESULTS
Hemoglobin: 10.2 g/dL
Creatinine: 1.3 mg/dL
"""

    sections = extract_clinical_sections(text)

    assert "PAST MEDICAL HISTORY" in sections
    assert "VITAL SIGNS" in sections
    assert "LABORATORY RESULTS" in sections

    conditions = extract_section_items(
        sections,
        "PAST MEDICAL HISTORY"
    )

    assert conditions == [
        "Hypertension",
        "Type 2 diabetes mellitus",
    ]

def test_link_section_concepts_to_values():
    """Test linking clinical concepts to nearby values."""

    section_entities = [
        {
            "text": "Blood pressure",
            "type": "Diagnostic_procedure",
            "start": 0,
            "end": 14,
            "section": "VITAL SIGNS",
        },
        {
            "text": "148/92 mmHg",
            "type": "Lab_value",
            "start": 16,
            "end": 27,
            "section": "VITAL SIGNS",
        },
        {
            "text": "Heart rate",
            "type": "Diagnostic_procedure",
            "start": 28,
            "end": 38,
            "section": "VITAL SIGNS",
        },
        {
            "text": "96 bpm",
            "type": "Lab_value",
            "start": 40,
            "end": 46,
            "section": "VITAL SIGNS",
        },
    ]

    linked = link_section_concepts_to_values(
        section_entities,
        "VITAL SIGNS"
    )

    assert len(linked) == 2

    assert linked[0]["name"] == "Blood pressure"
    assert linked[0]["value"] == "148/92 mmHg"

    assert linked[1]["name"] == "Heart rate"
    assert linked[1]["value"] == "96 bpm"