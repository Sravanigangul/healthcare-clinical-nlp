import re
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import spacy
nlp = spacy.blank("en")


from app.references import DRUG_LIST, DDI_REF


DOSE_UNIT = r'(?:mg|g|mcg|ml|l|units|IU|mEq|meq|pills|tablets|capsules)'

ROUTE_PAT = r'(?:PO|IV|IM|SC|SL|PR|topical|inhalation|nasal|ophthalmic|otic)'

FREQ_PAT = (
    r'(?:QD|BID|TID|QID|PRN|'
    r'once\s+daily|twice\s+daily|three\s+times\s+daily|four\s+times\s+daily|'
    r'once\s+a\s+day|twice\s+a\s+day|three\s+times\s+a\s+day|'
    r'daily|weekly|monthly|'
    r'every\s+\d+\s+(?:hours?|days?|weeks?|months?))'
)

DRUG_PAT = (
    r'\b('
    + '|'.join(re.escape(drug) for drug in DRUG_LIST)
    + r')\b'
)

MED_PAT = (
    DRUG_PAT
    + r'(?:[\s,]+(\d+(?:\.\d+)?)\s*(' + DOSE_UNIT + r'))?'
    + r'(?:[\s,]+(' + ROUTE_PAT + r'))?'
    + r'(?:[\s,]+(' + FREQ_PAT + r'))?'
)

LAB_PATTERNS = {
    "hemoglobin": r'\bhemoglobin\s+(?:was\s+|is\s+)?(\d+(?:\.\d+)?)\s*(g/dL)?',
    "creatinine": r'\bcreatinine\s+(?:was\s+|is\s+)?(\d+(?:\.\d+)?)\s*(mg/dL)?',
    "glucose": r'\bglucose\s+(?:was\s+|is\s+)?(\d+(?:\.\d+)?)\s*(mg/dL)?',
}

def extract_meds(text):
    seen = set()
    meds = []

    for match in re.finditer(MED_PAT, text, re.IGNORECASE):
        drug = match.group(1).lower()

        if drug not in seen:
            seen.add(drug)

            meds.append({
                "drug": drug,
                "dose": (match.group(2) or "").strip(),
                "unit": (match.group(3) or "").strip(),
                "route": (match.group(4) or "").upper(),
                "frequency": (match.group(5) or "").upper(),
            })

    return meds


def check_ddi(medication_list):
    med_names = {
        med["drug"].lower()
        for med in medication_list
    }

    flagged = []

    for drug1, drug2, severity, message in DDI_REF:
        if drug1 in med_names and drug2 in med_names:
            flagged.append({
                "drug1": drug1,
                "drug2": drug2,
                "severity": severity,
                "message": message,
            })

    return flagged

def extract_labs(text):
    """
    Extract common laboratory values from clinical text.
    """

    labs = []

    for lab_name, pattern in LAB_PATTERNS.items():

        for match in re.finditer(
            pattern,
            text,
            re.IGNORECASE
        ):

            labs.append({
                "name": lab_name,
                "value": match.group(1),
                "unit": match.group(2) or ""
            })

    return labs


def analyze_note(text):
    if not text or not text.strip():
        return {
            "medications": [],
            "drug_interactions": [],
        }

    meds = extract_meds(text)
    interactions = check_ddi(meds)

    return {
        "medications": meds,
        "drug_interactions": interactions,
    }

def create_patient_record():
    """
    Create the standard output structure for our clinical document.
    """

    patient_record = {
        "demographics": {
            "age":[],
            "sex":[]
        },

        "conditions":[],
        "symptoms":[],
        "medications":[],
        "labs":[],
        "vitals": [],
        "diagnostic_procedures":[],
        "therapeutic_procedures":[],
        "negated_findings":[],
        "icd10_candidates" :[],
        "drug_interactions":[],
        "summary":""
    }

    return patient_record
def load_ner_model():
    """
    Load the fine-tuned BioClinicalBERT NER model.
    """

    project_root = Path(__file__).resolve().parent.parent

    checkpoint_path = (
        project_root
        / "models"
        / "clinical_ner"
        / "checkpoint-216"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path
    )

    model = AutoModelForTokenClassification.from_pretrained(
        checkpoint_path
    )

    model.eval()

    return tokenizer, model

def extract_clinical_entities(text, tokenizer, model):
    """
    Extract clinical entities from a short clinical note
    using the fine-tuned BioClinicalBERT model.
    """

    #Convert text into tokens/numbers BERT understands
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation= True,
        max_length=512
    )

    #Turn off gradient calculations because we are not training
    with torch.no_grad():
        outputs = model(**inputs)

    #Get the most likely label for every token
    predicted_ids = torch.argmax(
        outputs.logits,
        dim=-1
    )[0]

    #Convert token IDs back into readable tokens
    tokens = tokenizer.convert_ids_to_tokens(
        inputs["input_ids"][0]
    )

    #Convert predicted label numbers into label names
    predictions = []

    for token, predicted_id in zip(tokens, predicted_ids):

        label = model.config.id2label[
            predicted_id.item()
        ]

        predictions.append({
            "token":token,
            "label":label
        })

    return predictions

def tokenize_clinical_text(text, tokenizer):
    """
    Tokenize clinical text using the same word-level structure
    used during BioClinicalBERT training.
    """

    # Step 1: Split the clinical text into words with spaCy
    doc = nlp(text)

    words = [
        token.text
        for token in doc
    ]

    # Step 2: Pass those words to the BioClinicalBERT tokenizer
    tokenized = tokenizer(
        words,
        is_split_into_words=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

    # Step 3: Map BERT tokens back to the original words
    word_ids = tokenized.word_ids(
        batch_index=0
    )

    return words, tokenized, word_ids

def predict_word_labels(text, tokenizer, model):
    """
    Predict one clinical NER label for each original word.
    """

    # Use the tokenization pipeline we just tested
    words, tokenized, word_ids = tokenize_clinical_text(
        text,
        tokenizer
    )

    # Run the trained BioClinicalBERT model
    with torch.no_grad():
        outputs = model(**tokenized)

    # Choose the highest-scoring label for every BERT token
    predicted_ids = torch.argmax(
        outputs.logits,
        dim=-1
    )[0]

    word_predictions = []

    previous_word_id = None

    for predicted_id, word_id in zip(
        predicted_ids,
        word_ids
    ):

        # Ignore [CLS] and [SEP]
        if word_id is None:
            continue

        # Only use the first WordPiece for each original word
        if word_id == previous_word_id:
            continue

        label = model.config.id2label[
            predicted_id.item()
        ]

        word_predictions.append({
            "word": words[word_id],
            "label": label
        })

        previous_word_id = word_id

    return word_predictions
def merge_bio_entities(word_predictions):
    """
    Merge word-level BIO predictions into complete clinical entities.
    """

    entities = []

    current_entity = None

    for prediction in word_predictions:

        word = prediction["word"]
        label = prediction["label"]

        # O means this word is not part of an entity
        if label == "O":

            if current_entity is not None:
                entities.append(current_entity)
                current_entity = None

            continue

        # Split B-Sign_symptom into:
        # prefix = B
        # entity_type = Sign_symptom
        prefix, entity_type = label.split("-", 1)

        if prefix == "B":

            # Save the previous entity first
            if current_entity is not None:
                entities.append(current_entity)

            # Start a new entity
            current_entity = {
                "text": word,
                "type": entity_type
            }

        elif prefix == "I":

            # Continue the existing entity
            if (
                current_entity is not None
                and current_entity["type"] == entity_type
            ):
                current_entity["text"] += " " + word

    # Save the final entity
    if current_entity is not None:
        entities.append(current_entity)

    return entities

def detect_negated_entities(text, entities):
    """
    Identify extracted entities that are preceded by
    common clinical negation terms.
    """

    negation_terms = [
        "no",
        "not",
        "denies",
        "denied",
        "without"
    ]

    negated_entities = []

    text_lower = text.lower()

    for entity in entities:

        entity_text = entity["text"].lower()

        entity_position = text_lower.find(entity_text)

        if entity_position == -1:
            continue

        text_before_entity = text_lower[:entity_position]

        words_before = text_before_entity.split()

        window = words_before[-5:]

        if any(
            term in window
            for term in negation_terms
        ):
            negated_entities.append(entity)

    return negated_entities

def add_entities_to_patient_record(patient_record, entities):
    """
    Add extracted clinical entities to the appropriate
    sections of the patient record.
    """

    for entity in entities:

        entity_text = entity["text"]
        entity_type = entity["type"]

        if entity_type == "Disease_disorder":
            patient_record["conditions"].append(entity_text)

        elif entity_type == "Sign_symptom":
            patient_record["symptoms"].append(entity_text)

        elif entity_type == "Lab_value":
            patient_record["labs"].append(entity_text)

        elif entity_type == "Diagnostic_procedure":
            patient_record["diagnostic_procedures"].append(entity_text)

        elif entity_type == "Therapeutic_procedure":
            patient_record["therapeutic_procedures"].append(entity_text)

        elif entity_type == "Age":
            patient_record["demographics"]["age"].append(entity_text)

        elif entity_type == "Sex":
            patient_record["demographics"]["sex"].append(entity_text)

    return patient_record

def build_patient_record(text, tokenizer, model):
    """
    Analyze a clinical note and populate the structured patient record.
    """


    patient_record = create_patient_record()

    # Extract medications
    medications = extract_meds(text)
    patient_record["medications"] = medications

    # Extract laboratory values
    labs = extract_labs(text)
    patient_record["labs"] = labs

    # Run clinical NER
    word_predictions = predict_word_labels(
        text,
        tokenizer,
        model
    )

    # Merge BIO labels into complete entities
    entities = merge_bio_entities(
        word_predictions
    )

    # Detect which entities are negated
    negated_entities = detect_negated_entities(
        text,
        entities
    )

    # Store negated findings
    patient_record["negated_findings"] = [
        entity["text"]
        for entity in negated_entities
    ]

    # Remove negated entities from active entities
    active_entities = [
        entity
        for entity in entities
        if entity not in negated_entities
    ]

    # Add only active entities to the patient record
    patient_record = add_entities_to_patient_record(
        patient_record,
        active_entities
    )

    return patient_record