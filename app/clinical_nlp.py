# pylint: disable=too-many-lines,too-many-locals
"""
Clinical NLP Pipeline
=====================

Core NER engine:
    Fine-tuned BioClinicalBERT

Supporting deterministic components:
    - Medication extraction
    - Laboratory value extraction
    - Negation detection
    - Drug-drug interaction lookup
    - Structured patient record generation

Long clinical documents are processed using overlapping
512-token windows with a stride of 128 tokens.
"""
# pylint: disable=too-many-lines

import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import spacy
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
)

from app.references import DRUG_LIST, DDI_REF


# ============================================================
# 1. BASIC NLP SETUP
# ============================================================

# We use spaCy only for word tokenization.
# No pretrained spaCy model is required.
nlp = spacy.blank("en")


# ============================================================
# 2. MEDICATION PATTERNS
# ============================================================

DOSE_UNIT = (
    r"(?:mg|g|mcg|ml|l|units|IU|mEq|meq|"
    r"pills|tablets|capsules)"
)

ROUTE_PAT = (
    r"(?:PO|IV|IM|SC|SL|PR|topical|inhalation|"
    r"nasal|ophthalmic|otic)"
)

FREQ_PAT = (
    r"(?:QD|BID|TID|QID|PRN|"
    r"once\s+daily|"
    r"twice\s+daily|"
    r"three\s+times\s+daily|"
    r"four\s+times\s+daily|"
    r"once\s+a\s+day|"
    r"twice\s+a\s+day|"
    r"three\s+times\s+a\s+day|"
    r"daily|weekly|monthly|"
    r"every\s+\d+\s+(?:hours?|days?|weeks?|months?))"
)


DRUG_PAT = (
    r"\b("
    + "|".join(
        re.escape(drug)
        for drug in DRUG_LIST
    )
    + r")\b"
)


MED_PAT = (
    DRUG_PAT
    + r"(?:[\s,]+(\d+(?:\.\d+)?)\s*("
    + DOSE_UNIT
    + r"))?"
    + r"(?:[\s,]+("
    + ROUTE_PAT
    + r"))?"
    + r"(?:[\s,]+("
    + FREQ_PAT
    + r"))?"
)


# ============================================================
# 3. LAB PATTERNS
# ============================================================

LAB_PATTERNS = {

    "hemoglobin": (
        r"\bhemoglobin\s+"
        r"(?:was\s+|is\s+)?"
        r"(\d+(?:\.\d+)?)\s*"
        r"(g/dL)?"
    ),

    "creatinine": (
        r"\bcreatinine\s+"
        r"(?:was\s+|is\s+)?"
        r"(\d+(?:\.\d+)?)\s*"
        r"(mg/dL)?"
    ),

    "glucose": (
        r"\bglucose\s+"
        r"(?:was\s+|is\s+)?"
        r"(\d+(?:\.\d+)?)\s*"
        r"(mg/dL)?"
    ),
}


# ============================================================
# 4. MEDICATION EXTRACTION
# ============================================================

def extract_meds(text):
    """
    Extract known medications and optional dose, unit,
    route and frequency information.

    This is a deterministic supporting layer.
    """

    seen = set()
    medications = []

    for match in re.finditer(
        MED_PAT,
        text,
        re.IGNORECASE
    ):

        drug = match.group(1).lower()

        # Avoid duplicate medication entries.
        if drug in seen:
            continue

        seen.add(drug)

        medications.append({
            "drug": drug,
            "dose": (
                match.group(2) or ""
            ).strip(),
            "unit": (
                match.group(3) or ""
            ).strip(),
            "route": (
                match.group(4) or ""
            ).upper(),
            "frequency": (
                match.group(5) or ""
            ).upper(),
        })

    return medications


# ============================================================
# 5. DRUG-DRUG INTERACTION CHECK
# ============================================================

def check_ddi(medication_list):
    """
    Check extracted medications against the small
    educational DDI reference table.

    This is NOT a comprehensive clinical DDI database.
    """

    medication_names = {
        medication["drug"].lower()
        for medication in medication_list
    }

    flagged_interactions = []

    for (
        drug1,
        drug2,
        severity,
        message
    ) in DDI_REF:

        if (
            drug1.lower() in medication_names
            and
            drug2.lower() in medication_names
        ):

            flagged_interactions.append({
                "drug1": drug1,
                "drug2": drug2,
                "severity": severity,
                "message": message,
            })

    return flagged_interactions


# ============================================================
# 6. LAB EXTRACTION
# ============================================================


def extract_labs(text):
    """Extract common laboratory values from clinical text."""

    lab_patterns = {
        "hemoglobin": r"\bhemoglobin\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*(g/dL)?",
        "creatinine": r"\bcreatinine\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*(mg/dL)?",
        "glucose": r"\bglucose\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*(mg/dL)?",
    }

    labs = []

    for lab_name, pattern in lab_patterns.items():
        matches = re.finditer(pattern, text, flags=re.IGNORECASE)

        for match in matches:
            labs.append(
                {
                    "lab": lab_name,
                    "value": match.group(1),
                    "unit": match.group(2) or "",
                }
            )

    return labs


# ============================================================
# 7. CREATE EMPTY PATIENT RECORD
# ============================================================

def create_patient_record():
    """
    Create the standard structured output used by
    the Clinical Document Review Assistant.
    """

    return {

        "demographics": {
            "age": [],
            "sex": [],
        },

        "conditions": [],

        "symptoms": [],

        "medications": [],

        "labs": [],

        "vitals": [],

        "diagnostic_procedures": [],

        "therapeutic_procedures": [],

        "negated_findings": [],

        "icd10_candidates": [],

        "drug_interactions": [],

        "entities": [],

        "summary": "",
    }


# ============================================================
# 8. LOAD BIOCLINICALBERT
# ============================================================

def load_ner_model(
    checkpoint_path=None
):
    """
    Load the fine-tuned BioClinicalBERT clinical NER model.

    If checkpoint_path is not supplied, the application
    looks for:

        models/clinical_ner_final/

    relative to the project root.
    """

    if checkpoint_path is None:

        project_root = (
            Path(__file__)
            .resolve()
            .parent
            .parent
        )

        checkpoint_path = (
            project_root
            / "models"
            / "clinical_ner_final"
        )

    checkpoint_path = Path(
        checkpoint_path
    )


    if not checkpoint_path.exists():

        raise FileNotFoundError(
            "\nFine-tuned clinical NER model was not found.\n\n"
            f"Expected location:\n{checkpoint_path}\n\n"
            "Reproduce/download the final BioClinicalBERT "
            "checkpoint and place it in this directory."
        )


    tokenizer = (
        AutoTokenizer.from_pretrained(
            checkpoint_path
        )
    )


    model = (
        AutoModelForTokenClassification
        .from_pretrained(
            checkpoint_path
        )
    )


    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    model.to(device)

    model.eval()


    return tokenizer, model


# ============================================================
# 9. OVERLAP-AWARE BIOCLINICALBERT NER
# ============================================================
# pylint: disable-next=too-many-locals
def predict_word_labels(
    text,
    tokenizer,
    model,
    max_length=512,
    stride=128
):
    """
    Predict one BIO label for each represented original word.

    Long documents are processed using overlapping chunks.

    Pipeline:

        clinical text
            ↓
        spaCy words
            ↓
        BioClinicalBERT WordPieces
            ↓
        512-token windows
            ↓
        128-token overlap
            ↓
        model logits
            ↓
        map back to original words
            ↓
        average overlapping logits
            ↓
        one BIO label per word
    """

    if not text or not text.strip():
        return []


    # --------------------------------------------------------
    # ORIGINAL WORD TOKENIZATION
    # --------------------------------------------------------

    doc = nlp(text)

    words = [
        token.text
        for token in doc
    ]

    word_offsets = [
        {
            "start": token.idx,
            "end": token.idx + len(token.text)
        }
        for token in doc
    ]


    if not words:
        return []


    # --------------------------------------------------------
    # WORDPIECE TOKENIZATION + OVERFLOW
    # --------------------------------------------------------

    tokenized = tokenizer(
        words,
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_tensors=None,
    )


    # --------------------------------------------------------
    # MODEL DEVICE
    # --------------------------------------------------------

    device = next(
        model.parameters()
    ).device


    model.eval()


    # word_id:
    #
    #     original spaCy word index
    #
    # value:
    #
    #     one or more 82-dimensional logit vectors
    #
    # Multiple vectors occur when a word appears in
    # overlapping chunks.

    word_logits = defaultdict(list)


    number_of_chunks = len(
        tokenized["input_ids"]
    )


    # --------------------------------------------------------
    # PROCESS EVERY CHUNK
    # --------------------------------------------------------

    with torch.no_grad():

        for chunk_idx in range(
            number_of_chunks
        ):

            input_ids = torch.tensor(
                [
                    tokenized[
                        "input_ids"
                    ][chunk_idx]
                ],
                dtype=torch.long,
                device=device,
            )


            attention_mask = torch.tensor(
                [
                    tokenized[
                        "attention_mask"
                    ][chunk_idx]
                ],
                dtype=torch.long,
                device=device,
            )


            model_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }


            # BERT models may provide token_type_ids.
            if "token_type_ids" in tokenized:

                model_inputs[
                    "token_type_ids"
                ] = torch.tensor(
                    [
                        tokenized[
                            "token_type_ids"
                        ][chunk_idx]
                    ],
                    dtype=torch.long,
                    device=device,
                )


            # ------------------------------------------------
            # BIOCLINICALBERT FORWARD PASS
            # ------------------------------------------------

            outputs = model(
                **model_inputs
            )


            # Shape:
            #
            # sequence length × number of labels
            #
            # In our trained model:
            #
            # sequence length × 82

            logits = (
                outputs
                .logits[0]
                .detach()
                .cpu()
                .numpy()
            )


            # ------------------------------------------------
            # MAP WORDPIECES → ORIGINAL WORDS
            # ------------------------------------------------

            word_ids = tokenized.word_ids(
                batch_index=chunk_idx
            )


            previous_word_id = None


            for token_idx, word_id in enumerate(
                word_ids
            ):

                # [CLS], [SEP], padding, etc.
                if word_id is None:
                    continue


                # Only use the first WordPiece belonging
                # to each original word.
                #
                # Example:
                #
                # palpitations
                #
                # p        ← use
                # ##al     ← ignore
                # ##pit    ← ignore
                # ##ations ← ignore

                if word_id == previous_word_id:
                    continue


                word_logits[
                    word_id
                ].append(
                    logits[token_idx]
                )


                previous_word_id = word_id


    # --------------------------------------------------------
    # RECONSTRUCT ORIGINAL WORD PREDICTIONS
    # --------------------------------------------------------

    predictions = []


    for word_id, word in enumerate(
        words
    ):

        # Certain whitespace/newline tokens may not produce
        # a WordPiece.
        if word_id not in word_logits:
            continue


        # Average predictions from overlapping windows.
        average_logits = np.mean(
            word_logits[word_id],
            axis=0
        )


        predicted_id = int(
            np.argmax(
                average_logits
            )
        )


        predicted_label = (
            model.config.id2label[
                predicted_id
            ]
        )


        # Convert logits to probabilities so the application
        # can display a model confidence value.
        probabilities = torch.softmax(
            torch.tensor(
                average_logits,
                dtype=torch.float32
            ),
            dim=-1,
        )


        confidence = float(
            probabilities[
                predicted_id
            ].item()
        )


        predictions.append({

            "word": word,

            "word_id": word_id,
            "start": word_offsets[word_id]["start"],

            "end": word_offsets[word_id]["end"],

            "label": predicted_label,

            "confidence": round(
                confidence,
                4
            ),
        })


    return predictions


# ============================================================
# 10. MERGE BIO LABELS INTO COMPLETE ENTITIES
# ============================================================


def merge_bio_entities(word_predictions):
    # pylint: disable=unsupported-assignment-operation
    # pylint: disable=unsubscriptable-object
    # pylint: disable=too-many-branches
    """
    Merge word-level BIO predictions into complete clinical entities.

    Example:
        chest -> B-Sign_symptom
        pain  -> I-Sign_symptom

    becomes:
        {
            "text": "chest pain",
            "type": "Sign_symptom",
            "confidence": 0.92
        }
    """

    entities = []

    current_entity = None

    for prediction in word_predictions:

        word = prediction["word"]
        label = prediction["label"]
        confidence = prediction.get("confidence", 0.0)
        start = prediction.get("start")
        end = prediction.get("end")

        # ----------------------------------------------------
        # O = word is outside an entity
        # ----------------------------------------------------

        if label == "O":

            if isinstance(current_entity, dict):

                scores = current_entity.pop("_scores", [])

                if scores:
                    current_entity["confidence"] = round(
                        sum(scores) / len(scores),
                        4
                    )
                else:
                    current_entity["confidence"] = 0.0

                entities.append(current_entity)

            current_entity = None

            continue

        # ----------------------------------------------------
        # Safety check
        # ----------------------------------------------------

        if "-" not in label:
            continue

        prefix, entity_type = label.split("-", 1)

        # ----------------------------------------------------
        # B = beginning of a new entity
        # ----------------------------------------------------

        if prefix == "B":

            # Save previous entity first
            if isinstance(current_entity, dict):

                scores = current_entity.pop("_scores", [])

                if scores:
                    current_entity["confidence"] = round(
                        sum(scores) / len(scores),
                        4
                    )
                else:
                    current_entity["confidence"] = 0.0

                entities.append(current_entity)

            # Start new entity
            current_entity = {
                "text": word,
                "type": entity_type,
                "start": start,
                "end": end,
                "_scores": [confidence]
            }

        # ----------------------------------------------------
        # I = continuation of an entity
        # ----------------------------------------------------

        elif prefix == "I":

            if (
                isinstance(current_entity, dict)
                and current_entity.get("type") == entity_type
            ):

                current_entity["text"] += " " + word

                current_entity["end"] = end

                current_entity["_scores"].append(
                    confidence
                )

            else:

                # BIO repair:
                #
                # If the model predicts I-X but there is no
                # matching B-X before it, treat this I-X as
                # the beginning of a new entity.

                if isinstance(current_entity, dict):

                    scores = current_entity.pop(
                        "_scores",
                        []
                    )

                    if scores:
                        current_entity["confidence"] = round(
                            sum(scores) / len(scores),
                            4
                        )
                    else:
                        current_entity["confidence"] = 0.0

                    entities.append(
                        current_entity
                    )

                current_entity = {
                    "text": word,
                    "type": entity_type,
                    "start": start,
                    "end": end,
                    "_scores": [confidence]
                }

    # --------------------------------------------------------
    # Save the final entity
    # --------------------------------------------------------

    if isinstance(current_entity, dict):

        scores = current_entity.pop(
            "_scores",
            []
        )

        if scores:
            current_entity["confidence"] = round(
                sum(scores) / len(scores),
                4
            )
        else:
            current_entity["confidence"] = 0.0

        entities.append(
            current_entity
        )

    return entities
# ============================================================
# 11. NEGATION DETECTION
# ============================================================

def detect_negated_entities(
    text,
    entities,
    window_size=5
):
    """
    Identify entities preceded by common clinical
    negation expressions.

    This is a lightweight rule-based supporting layer,
    not a full clinical negation model.
    """

    negation_terms = {
        "no",
        "not",
        "denies",
        "denied",
        "without",
        "negative",
    }


    negated_entities = []


    text_lower = text.lower()


    for entity in entities:

        entity_text = (
            entity["text"]
            .lower()
        )


        entity_position = (
            text_lower.find(
                entity_text
            )
        )


        if entity_position == -1:
            continue


        text_before_entity = (
            text_lower[
                :entity_position
            ]
        )


        words_before = (
            text_before_entity
            .split()
        )


        window = words_before[
            -window_size:
        ]


        if any(
            term in window
            for term in negation_terms
        ):

            negated_entity = (
                entity.copy()
            )

            negated_entity[
                "negated"
            ] = True

            negated_entities.append(
                negated_entity
            )


    return negated_entities


# ============================================================
# 12. ADD BIOCLINICALBERT ENTITIES TO PATIENT RECORD
# ============================================================

def add_entities_to_patient_record(
    patient_record,
    entities
):
    """
    Map selected BioClinicalBERT entity categories into
    clinically useful patient-record sections.

    The complete raw entity list is still retained
    separately under patient_record["entities"].
    """

    for entity in entities:

        entity_text = entity[
            "text"
        ]

        entity_type = entity[
            "type"
        ]


        if entity_type == "Disease_disorder":

            patient_record[
                "conditions"
            ].append(
                entity_text
            )


        elif entity_type == "Sign_symptom":

            patient_record[
                "symptoms"
            ].append(
                entity_text
            )


        elif entity_type == "Diagnostic_procedure":

            patient_record[
                "diagnostic_procedures"
            ].append(
                entity_text
            )


        elif entity_type == "Therapeutic_procedure":

            patient_record[
                "therapeutic_procedures"
            ].append(
                entity_text
            )


        elif entity_type == "Age":

            patient_record[
                "demographics"
            ][
                "age"
            ].append(
                entity_text
            )


        elif entity_type == "Sex":

            patient_record[
                "demographics"
            ][
                "sex"
            ].append(
                entity_text
            )


    return patient_record


# ============================================================
# 13. REMOVE DUPLICATES
# ============================================================

def deduplicate_list(values):
    """
    Remove duplicate strings while preserving order.
    """

    seen = set()

    result = []


    for value in values:

        key = (
            value.lower()
            if isinstance(value, str)
            else str(value)
        )


        if key not in seen:

            seen.add(key)

            result.append(
                value
            )


    return result

def extract_vitals_from_entities(entities):
    """
    Build structured vital signs from BioClinicalBERT entities.

    BioClinicalBERT identifies the clinical concept and its nearby
    Lab_value. This function links recognized vital concepts with
    their following value.
    """

    vital_names = {
        "blood pressure",
        "heart rate",
        "temperature",
        "respiratory rate",
        "oxygen saturation",
        "spo2",
    }

    vitals = []

    for index, entity in enumerate(entities):
        entity_text = entity["text"].strip().lower()
        # Is this entity a recognized vital concept?
        if entity_text not in vital_names:
            continue
        #Look at the next model-extracted entity
        if index + 1 >= len(entities):
            continue
        next_entity = entities[index + 1]
        #The value should have been identifies by BERT as Lab_value.
        if next_entity["type"] != "Lab_value":
            continue

        vitals.append(
            {
                "name" : entity_text,
                "value": next_entity["text"],
                "concept_confidence": entity.get("confidence"),
                "value_confidence": next_entity.get("confidence"),
            }
        )

def extract_clinical_sections(text):
    """
    Split a clinical document into sections based on section headings.

    Returns:
        dict: Section name -> section text
    """

    sections = {}
    current_section = "UNSECTIONED"
    current_lines = []

    for line in text.splitlines():

        cleaned_line = line.strip()

        if not cleaned_line:
            continue

        # A short, uppercase line is treated as a possible section heading.
        is_heading = (
            cleaned_line.isupper()
            and len(cleaned_line.split()) <= 6
        )

        if is_heading:

            # Save the previous section.
            if current_lines:
                sections[current_section] = "\n".join(current_lines)

            current_section = cleaned_line
            current_lines = []

        else:
            current_lines.append(cleaned_line)

    # Save the final section.
    if current_lines:
        sections[current_section] = "\n".join(current_lines)

    return sections
def extract_clinical_section_spans(text):
    """
    Identify clinical sections while preserving their
    positions in the original document.

    Returns:
        list: Section dictionaries containing the section
        name, text, start offset, and end offset.
    """

    sections = []

    lines = text.splitlines(keepends=True)

    current_section = None
    current_start = None
    current_lines = []

    position = 0

    for line in lines:
        cleaned_line = line.strip()

        is_heading = (
            cleaned_line
            and cleaned_line.isupper()
            and len(cleaned_line.split()) <= 6
        )

        if is_heading:
            if current_section is not None:
                section_text = "".join(
                    current_lines
                ).strip()

                sections.append({
                    "section": current_section,
                    "text": section_text,
                    "start": current_start,
                    "end": position,
                })

            current_section = cleaned_line
            current_start = position + len(line)
            current_lines = []

        elif current_section is not None:
            current_lines.append(line)

        position += len(line)

    if current_section is not None:
        section_text = "".join(
            current_lines
        ).strip()

        sections.append({
            "section": current_section,
            "text": section_text,
            "start": current_start,
            "end": len(text),
        })

    return sections

def assign_entities_to_sections(
    entities,
    section_spans
):
    """
    Assign full-document entities to their clinical sections
    using character offsets.

    Args:
        entities:
            Entities extracted from the full clinical document.

        section_spans:
            Clinical sections with global start/end offsets.

    Returns:
        list: Entities with section information added.
    """

    section_entities = []

    for entity in entities:

        entity_start = entity.get("start")
        entity_end = entity.get("end")

        if entity_start is None or entity_end is None:
            continue

        entity_with_section = entity.copy()

        for section in section_spans:

            section_start = section["start"]
            section_end = section["end"]

            if (
                entity_start >= section_start
                and entity_end <= section_end
            ):
                entity_with_section["section"] = (
                    section["section"]
                )
                break

        section_entities.append(
            entity_with_section
        )

    return section_entities

def extract_section_items(sections, section_name):
    """
    Extract individual line-based items from a clinical section.

    Useful for list-like sections such as:
        PAST MEDICAL HISTORY
        ALLERGIES
        MEDICATIONS

    Each non-empty line is preserved as a separate item.
    """

    section_text = sections.get(
        section_name,
        ""
    )

    if not section_text:
        return []

    items = [
        line.strip()
        for line in section_text.splitlines()
        if line.strip()
    ]

    return items

def extract_section_entities(
    sections,
    tokenizer,
    model
):
    """
    Run BioClinicalBERT on each clinical section separately
    and preserve the section associated with every entity.
    """

    section_entities = []

    for section_name, section_text in sections.items():

        # Run BioClinicalBERT on this section.
        word_predictions = predict_word_labels(
            section_text,
            tokenizer,
            model,
            max_length=512,
            stride=128,
        )

        # Convert BIO token predictions into complete entities.
        entities = merge_bio_entities(
            word_predictions
        )

        # Preserve the section for each entity.
        for entity in entities:

            entity_with_section = entity.copy()

            entity_with_section[
                "section"
            ] = section_name

            section_entities.append(
                entity_with_section
            )

    return section_entities

def link_section_concepts_to_values(
    section_entities,
    section_name,
    concept_types=None,
    value_types=None,
    max_distance=20,
):
    """
    Link clinical concepts to nearby values within the same section.

    Example:
        Blood pressure -> 148/92 mmHg
        Heart rate     -> 96 bpm
    """

    if concept_types is None:
        concept_types = {"Diagnostic_procedure"}

    if value_types is None:
        value_types = {"Lab_value"}

    # Keep only entities belonging to the requested section.
    entities = [
        entity
        for entity in section_entities
        if entity.get("section") == section_name
    ]

    # Keep entities in document order.
    entities.sort(
        key=lambda entity: entity.get("start", 0)
    )

    linked_values = []

    for index, entity in enumerate(entities):

        # We only start a pair from a clinical concept.
        if entity.get("type") not in concept_types:
            continue

        concept_end = entity.get("end")

        if concept_end is None:
            continue

        # Search forward for the nearest value.
        for candidate in entities[index + 1:]:

            candidate_start = candidate.get("start")

            if candidate_start is None:
                continue

            distance = candidate_start - concept_end

            # Candidate is too far away.
            if distance > max_distance:
                break

            if candidate.get("type") in value_types:

                linked_values.append({
                    "name": entity["text"],
                    "value": candidate["text"],
                    "section": section_name,
                    "concept_confidence": entity.get("confidence"),
                    "value_confidence": candidate.get("confidence"),
                    "start": entity["start"],
                    "end": candidate["end"],
                })

                break

    return linked_values
# ============================================================
# 14. BUILD COMPLETE PATIENT RECORD
# ============================================================

def build_patient_record(
    text,
    tokenizer,
    model
):
    """
    Run the complete Clinical Document Review pipeline.

    BioClinicalBERT runs once on the full clinical document.
    Extracted entities are then assigned to clinical sections
    using global character offsets.
    """

    patient_record = create_patient_record()

    if not text or not text.strip():
        return patient_record

    # --------------------------------------------------------
    # A. MEDICATION EXTRACTION
    # --------------------------------------------------------

    medications = extract_meds(text)
    patient_record["medications"] = medications

    # --------------------------------------------------------
    # B. LAB EXTRACTION - RULE-BASED FALLBACK
    # --------------------------------------------------------

    structured_labs = extract_labs(text)
    patient_record["labs"] = structured_labs

    # --------------------------------------------------------
    # C. DRUG INTERACTIONS
    # --------------------------------------------------------

    patient_record["drug_interactions"] = check_ddi(
        medications
    )

    # --------------------------------------------------------
    # D. CLINICAL SECTION DETECTION
    # --------------------------------------------------------

    sections = extract_clinical_sections(text)

    section_spans = extract_clinical_section_spans(
        text
    )

    # Conditions from the structured medical-history section.
    history_conditions = extract_section_items(
        sections,
        "PAST MEDICAL HISTORY"
    )

    # --------------------------------------------------------
    # E. BIOCLINICALBERT - SINGLE FULL-DOCUMENT PASS
    # --------------------------------------------------------

    word_predictions = predict_word_labels(
        text,
        tokenizer,
        model,
        max_length=512,
        stride=128,
    )

    entities = merge_bio_entities(
        word_predictions
    )

    patient_record["entities"] = entities

    # --------------------------------------------------------
    # F. ASSIGN FULL-DOCUMENT ENTITIES TO SECTIONS
    # --------------------------------------------------------

    section_entities = assign_entities_to_sections(
        entities,
        section_spans
    )

    # --------------------------------------------------------
    # G. SECTION-AWARE VALUE LINKING
    # --------------------------------------------------------

    linked_vitals = link_section_concepts_to_values(
        section_entities,
        "VITAL SIGNS"
    )

    linked_labs = link_section_concepts_to_values(
        section_entities,
        "LABORATORY RESULTS"
    )

    patient_record["vitals"] = linked_vitals

    # Prefer model-based section-aware labs when available.
    # Keep regex extraction as a fallback.
    if linked_labs:
        patient_record["labs"] = linked_labs

    # --------------------------------------------------------
    # H. NEGATION
    # --------------------------------------------------------

    negated_entities = detect_negated_entities(
        text,
        entities
    )

    patient_record["negated_findings"] = [
        {
            "text": entity["text"],
            "type": entity["type"],
            "confidence": entity.get("confidence"),
        }
        for entity in negated_entities
    ]

    negated_keys = {
        (
            entity["text"].lower(),
            entity["type"],
        )
        for entity in negated_entities
    }

    active_entities = [
        entity
        for entity in entities
        if (
            entity["text"].lower(),
            entity["type"],
        )
        not in negated_keys
    ]

    # --------------------------------------------------------
    # I. MAP ACTIVE ENTITIES INTO STRUCTURED RECORD
    # --------------------------------------------------------

    patient_record = add_entities_to_patient_record(
        patient_record,
        active_entities
    )

    # --------------------------------------------------------
    # J. ADD SECTION-BASED CONDITIONS
    # --------------------------------------------------------

    patient_record["conditions"].extend(
        history_conditions
    )

    # --------------------------------------------------------
    # K. DEDUPLICATE SIMPLE OUTPUTS
    # --------------------------------------------------------

    patient_record["conditions"] = deduplicate_list(
        patient_record["conditions"]
    )

    patient_record["symptoms"] = deduplicate_list(
        patient_record["symptoms"]
    )

    patient_record["diagnostic_procedures"] = deduplicate_list(
        patient_record["diagnostic_procedures"]
    )

    patient_record["therapeutic_procedures"] = deduplicate_list(
        patient_record["therapeutic_procedures"]
    )

    patient_record["demographics"]["age"] = deduplicate_list(
        patient_record["demographics"]["age"]
    )

    patient_record["demographics"]["sex"] = deduplicate_list(
        patient_record["demographics"]["sex"]
    )

    return patient_record

def analyze_note(
    text,
    tokenizer=None,
    model=None
):
    """
    Analyze a clinical note.

    If a BioClinicalBERT tokenizer/model is supplied,
    run the complete pipeline.

    Otherwise return only the deterministic medication,
    laboratory and DDI components.
    """

    if not text or not text.strip():

        if tokenizer is not None and model is not None:
            return create_patient_record()

        return {
            "medications": [],
            "labs": [],
            "drug_interactions": [],
        }


    # Full model pipeline
    if (
        tokenizer is not None
        and
        model is not None
    ):

        return build_patient_record(
            text,
            tokenizer,
            model
        )


    # Supporting rules only
    medications = extract_meds(
        text
    )


    return {

        "medications": medications,

        "labs": extract_labs(
            text
        ),

        "drug_interactions": check_ddi(
            medications
        ),
    }


# ============================================================
# 16. OPTIONAL QUICK TEST
# ============================================================

if __name__ == "__main__":

    example_text = """
    A 54-year-old female presented with chest pain and
    shortness of breath. She denies fever.

    Creatinine was 1.4 mg/dL and hemoglobin was
    10.8 g/dL.

    The patient takes aspirin 81 mg PO daily.
    """


    print(
        "\nClinical NLP supporting-rule test\n"
    )


    print(
        analyze_note(
            example_text
        )
    )


    print(
        "\nTo run BioClinicalBERT NER, place the final "
        "checkpoint in models/clinical_ner_final/."
    )
