from app.clinical_nlp import (
    extract_meds,
    check_ddi,
    analyze_note,
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