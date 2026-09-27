import html

import gradio as gr  # type: ignore[import-not-found]

from app.clinical_nlp import (
    analyze_note,
    build_patient_record,
    load_ner_model,
)

from app.icd_linker import (
    add_icd_candidates,
    load_icd10_codes,
    load_icd_embeddings,
    load_sapbert_model,
)

from app.drug_normalizer import enrich_patient_medications

from app.pdf_parser import (
    extract_pdf_pages,
    combine_pdf_text,
    build_page_spans,
)


# ============================================================
# MODEL AND TERMINOLOGY PATHS
# ============================================================

NER_MODEL_PATH = "models/clinical_ner_v2_5epochs_best"
ICD_CODES_PATH = "data/terminology/icd10cm-codes-2026.txt"
ICD_EMBEDDINGS_PATH = "models/icd10_sapbert_embeddings.pt"


# ============================================================
# LOAD MODELS ONCE
# ============================================================

print("Loading BioClinicalBERT NER model...")

ner_tokenizer, ner_model = load_ner_model(
    NER_MODEL_PATH
)

print("Loading ICD-10-CM terminology...")

icd_records = load_icd10_codes(
    ICD_CODES_PATH
)

print("Loading precomputed ICD-10 embeddings...")

icd_embeddings = load_icd_embeddings(
    ICD_EMBEDDINGS_PATH
)

print("Loading SapBERT concept-linking model...")

sapbert_tokenizer, sapbert_model = load_sapbert_model()

print("Clinical NLP models loaded.")


# ============================================================
# ENRICHMENT PIPELINE
# ============================================================

def enrich_patient_record(patient_record):
    """Add ICD-10, RxNorm, and DailyMed enrichment."""

    patient_record = add_icd_candidates(
        patient_record=patient_record,
        icd_records=icd_records,
        icd_embeddings=icd_embeddings,
        tokenizer=sapbert_tokenizer,
        model=sapbert_model,
        top_k=5,
    )

    patient_record = enrich_patient_medications(
        patient_record
    )

    return patient_record


# ============================================================
# ANALYSIS PIPELINE
# ============================================================

def analyze_document(text, pdf_file):
    """Analyze pasted text or an uploaded clinical PDF."""

    if pdf_file is not None:

        pages = extract_pdf_pages(pdf_file)

        if not pages:
            return {
                "error": "No extractable text was found in the PDF."
            }

        document_text = combine_pdf_text(pages)
        page_spans = build_page_spans(pages)

        patient_record = build_patient_record(
            document_text,
            ner_tokenizer,
            ner_model,
            page_spans=page_spans,
        )

        return enrich_patient_record(patient_record)

    if text and text.strip():

        patient_record = analyze_note(
            text,
            tokenizer=ner_tokenizer,
            model=ner_model,
        )

        return enrich_patient_record(patient_record)

    return {
        "error": "Please paste a clinical note or upload a PDF."
    }


# ============================================================
# DISPLAY HELPERS
# ============================================================

def safe(value):
    """Escape text before displaying it as HTML."""

    if value is None:
        return ""

    return html.escape(str(value))


def page_badge(page):
    """Create a page provenance badge."""

    if not page:
        return ""

    return (
        f'<span class="page-badge">'
        f'Page {safe(page)}'
        f'</span>'
    )


def confidence_badge(confidence):
    """Create a model-confidence badge."""

    if confidence is None:
        return ""

    try:
        percent = float(confidence) * 100
    except (TypeError, ValueError):
        return ""

    return (
        f'<span class="confidence-badge">'
        f'{percent:.0f}% confidence'
        f'</span>'
    )


def empty_state(message):
    return (
        '<div class="empty-state">'
        f'{safe(message)}'
        '</div>'
    )


# ============================================================
# PATIENT OVERVIEW
# ============================================================

def render_overview(record):

    demographics = record.get("demographics", {})

    ages = demographics.get("age", [])
    sexes = demographics.get("sex", [])

    age = ages[0] if ages else "Not extracted"
    sex = sexes[0] if sexes else "Not extracted"

    conditions = len(record.get("conditions", []))
    medications = len(record.get("medications", []))
    labs = len(record.get("labs", []))
    vitals = len(record.get("vitals", []))

    return f"""
    <div class="section-card">

        <div class="section-title">
            Patient Overview
        </div>

        <div class="metric-grid">

            <div class="metric-card">
                <div class="metric-label">Age</div>
                <div class="metric-value">{safe(age)}</div>
            </div>

            <div class="metric-card">
                <div class="metric-label">Sex</div>
                <div class="metric-value">{safe(sex)}</div>
            </div>

            <div class="metric-card">
                <div class="metric-label">Conditions</div>
                <div class="metric-value">{conditions}</div>
            </div>

            <div class="metric-card">
                <div class="metric-label">Medications</div>
                <div class="metric-value">{medications}</div>
            </div>

            <div class="metric-card">
                <div class="metric-label">Labs</div>
                <div class="metric-value">{labs}</div>
            </div>

            <div class="metric-card">
                <div class="metric-label">Vitals</div>
                <div class="metric-value">{vitals}</div>
            </div>

        </div>

    </div>
    """


# ============================================================
# CONDITIONS + ICD-10
# ============================================================

def render_conditions(record):

    candidates = record.get(
        "icd10_candidates",
        [],
    )

    if not candidates:
        return empty_state(
            "No conditions or ICD-10-CM candidates were extracted."
        )

    cards = []

    for condition_entry in candidates:

        condition = condition_entry.get(
            "condition",
            "Unknown condition",
        )

        matches = condition_entry.get(
            "candidates",
            [],
        )

        candidate_rows = []

        for candidate in matches[:5]:

            code = candidate.get(
                "code",
                ""
            )

            description = (
                candidate.get("description")
                or candidate.get("label")
                or ""
            )

            similarity = candidate.get(
                "similarity"
            )

            similarity_text = ""

            if similarity is not None:
                try:
                    similarity_text = (
                        f"{float(similarity):.3f}"
                    )
                except (TypeError, ValueError):
                    similarity_text = safe(
                        similarity
                    )

            candidate_rows.append(
                f"""
                <div class="candidate-row">
                    <span class="code-badge">
                        {safe(code)}
                    </span>

                    <span class="candidate-description">
                        {safe(description)}
                    </span>

                    <span class="similarity">
                        Similarity {safe(similarity_text)}
                    </span>
                </div>
                """
            )

        if not candidate_rows:
            candidate_rows.append(
                empty_state(
                    "No ICD-10-CM candidates available."
                )
            )

        cards.append(
            f"""
            <div class="item-card">

                <div class="item-title">
                    {safe(condition)}
                </div>

                <div class="small-label">
                    Ranked ICD-10-CM candidates
                </div>

                {''.join(candidate_rows)}

            </div>
            """
        )

    return "".join(cards)


# ============================================================
# MEDICATIONS
# ============================================================

def render_medications(record):

    medications = record.get(
        "medications",
        [],
    )

    if not medications:
        return empty_state(
            "No medications were extracted."
        )

    cards = []

    for medication in medications:

        drug = medication.get(
            "drug",
            "Unknown medication",
        )

        dose = medication.get(
            "dose",
            ""
        )

        unit = medication.get(
            "unit",
            ""
        )

        frequency = medication.get(
            "frequency",
            ""
        )

        route = medication.get(
            "route",
            ""
        )

        rxcui = medication.get(
            "rxcui",
            "Not available",
        )

        normalized = medication.get(
            "normalized_name",
            ""
        )

        page = medication.get(
            "page"
        )

        confidence = medication.get(
            "medication_confidence"
        )

        medication_line = " ".join(
            str(value)
            for value in [
                dose,
                unit,
                frequency,
                route,
            ]
            if value
        )

        safety = medication.get(
            "safety_evidence"
        )

        safety_html = ""

        if safety:

            title = safety.get(
                "label_title",
                ""
            )

            source = safety.get(
                "source",
                "DailyMed",
            )

            setid = safety.get(
                "setid",
                ""
            )

            evidence = safety.get(
                "evidence",
                {},
            )

            evidence_names = [
                key.replace("_", " ").title()
                for key in evidence.keys()
            ]

            evidence_text = (
                ", ".join(evidence_names)
                if evidence_names
                else "Label retrieved"
            )

            safety_html = f"""
            <div class="evidence-box">

                <div class="small-label">
                    Medication Safety Evidence
                </div>

                <div>
                    {safe(evidence_text)}
                </div>

                <div class="source-text">
                    Source: {safe(source)}
                </div>

                <div class="source-text">
                    Label: {safe(title)}
                </div>

                <div class="source-text">
                    Set ID: {safe(setid)}
                </div>

            </div>
            """

        cards.append(
            f"""
            <div class="item-card">

                <div class="item-title">
                    {safe(drug)}
                </div>

                <div class="medication-details">
                    {safe(medication_line)}
                </div>

                <div class="badge-row">
                    {page_badge(page)}
                    {confidence_badge(confidence)}
                </div>

                <div class="normalization-box">
                    <b>RxNorm:</b>
                    {safe(normalized or "Not available")}
                    &nbsp; | &nbsp;
                    <b>RxCUI:</b>
                    {safe(rxcui)}
                </div>

                {safety_html}

            </div>
            """
        )

    return "".join(cards)


# ============================================================
# LABS / VITALS
# ============================================================

def render_measurements(items, empty_message):

    if not items:
        return empty_state(
            empty_message
        )

    rows = []

    for item in items:

        name = item.get(
            "name",
            ""
        )

        value = item.get(
            "value",
            ""
        )

        page = item.get(
            "page"
        )

        confidence = item.get(
            "value_confidence"
        )

        rows.append(
            f"""
            <div class="measurement-row">

                <div>
                    <div class="measurement-name">
                        {safe(name)}
                    </div>

                    <div class="measurement-value">
                        {safe(value)}
                    </div>
                </div>

                <div class="badge-row">
                    {page_badge(page)}
                    {confidence_badge(confidence)}
                </div>

            </div>
            """
        )

    return "".join(rows)


# ============================================================
# ALLERGIES
# ============================================================

def render_allergies(record):
    """Render extracted allergies cleanly."""

    allergies = record.get(
        "allergies",
        {},
    )

    status = allergies.get(
        "status",
        "unknown",
    )

    items = allergies.get(
        "items",
        [],
    )

    if not items:
        return f"""
        <div class="item-card">
            <div class="small-label">
                Allergy status
            </div>
            <div>
                {safe(status)}
            </div>
        </div>
        """

    cards = []

    for item in items:

        if isinstance(item, dict):

            # Current allergy objects use source_text.
            text = (
                item.get("source_text")
                or item.get("text")
                or item.get("allergy")
                or item.get("substance")
                or "Unknown allergy"
            )

            page = item.get("page")
            confidence = item.get("confidence")

        else:
            text = str(item)
            page = None
            confidence = None

        cards.append(
            f"""
            <div class="item-card">

                <div class="item-title">
                    {safe(text)}
                </div>

                <div class="badge-row">
                    {page_badge(page)}
                    {confidence_badge(confidence)}
                </div>

            </div>
            """
        )

    return "".join(cards)


# ============================================================
# NEGATED FINDINGS
# ============================================================

def render_negated_findings(record):
    """Render clinically relevant negated findings."""

    findings = record.get(
        "negated_findings",
        [],
    )

    if not findings:
        return empty_state(
            "No negated findings were extracted."
        )

    cleaned = []

    # Entity types that make sense as clinical negated findings.
    allowed_types = {
        "Sign_symptom",
        "Disease_disorder",
        "Biological_structure",
    }

    for finding in findings:

        if isinstance(finding, dict):

            text = finding.get(
                "text",
                ""
            ).strip()

            entity_type = finding.get(
                "type",
                ""
            )

            # Remove non-clinical entities such as
            # "Patient" / Detailed_description.
            if entity_type not in allowed_types:
                continue

        else:
            text = str(finding).strip()

        if text:
            cleaned.append(text)

    # Combine the adjacent entities "chest" + "pain".
    combined = []
    index = 0

    while index < len(cleaned):

        if (
            index + 1 < len(cleaned)
            and cleaned[index].lower() == "chest"
            and cleaned[index + 1].lower() == "pain"
        ):
            combined.append("chest pain")
            index += 2
            continue

        combined.append(
            cleaned[index]
        )

        index += 1

    # Remove duplicates while preserving order.
    unique_findings = list(
        dict.fromkeys(combined)
    )

    if not unique_findings:
        return empty_state(
            "No clinically relevant negated findings were extracted."
        )

    badges = []

    for text in unique_findings:

        badges.append(
            f"""
            <span class="finding-badge">
                ✕ {safe(text)}
            </span>
            """
        )

    return (
        '<div class="finding-container">'
        + " ".join(badges)
        + "</div>"
    )


# ============================================================
# DASHBOARD OUTPUT
# ============================================================

def run_dashboard(text, pdf_file):

    record = analyze_document(
        text,
        pdf_file,
    )

    if "error" in record:

        error = (
            f'<div class="error-box">'
            f'{safe(record["error"])}'
            f'</div>'
        )

        return (
            error,
            "",
            "",
            "",
            "",
            "",
            "",
            record,
        )

    overview = render_overview(
        record
    )

    conditions = render_conditions(
        record
    )

    medications = render_medications(
        record
    )

    labs = render_measurements(
        record.get("labs", []),
        "No laboratory results were extracted.",
    )

    vitals = render_measurements(
        record.get("vitals", []),
        "No vital signs were extracted.",
    )

    allergies = render_allergies(
        record
    )

    negated = render_negated_findings(
        record
    )

    return (
        overview,
        conditions,
        medications,
        labs,
        vitals,
        allergies,
        negated,
        record,
    )


# ============================================================
# CUSTOM CSS
# ============================================================

CSS = """
.gradio-container {
    max-width: 1400px !important;
    margin: auto !important;
}

.hero {
    padding: 28px;
    border: 1px solid var(--border-color-primary);
    border-radius: 18px;
    margin-bottom: 18px;
}

.hero-title {
    font-size: 30px;
    font-weight: 700;
    margin-bottom: 8px;
}

.hero-subtitle {
    opacity: 0.75;
    font-size: 15px;
    line-height: 1.5;
}

.section-card {
    border: 1px solid var(--border-color-primary);
    border-radius: 16px;
    padding: 18px;
    margin-bottom: 14px;
}

.section-title {
    font-size: 19px;
    font-weight: 700;
    margin-bottom: 14px;
}

.metric-grid {
    display: grid;
    grid-template-columns: repeat(
        auto-fit,
        minmax(130px, 1fr)
    );
    gap: 10px;
}

.metric-card {
    border: 1px solid var(--border-color-primary);
    border-radius: 12px;
    padding: 12px;
}

.metric-label {
    font-size: 12px;
    opacity: 0.65;
    margin-bottom: 5px;
}

.metric-value {
    font-size: 18px;
    font-weight: 700;
}

.item-card {
    border: 1px solid var(--border-color-primary);
    border-radius: 12px;
    padding: 14px;
    margin-bottom: 10px;
}

.item-title {
    font-size: 16px;
    font-weight: 700;
    margin-bottom: 6px;
}

.small-label {
    font-size: 12px;
    font-weight: 600;
    opacity: 0.7;
    margin-bottom: 7px;
}

.candidate-row {
    display: grid;
    grid-template-columns: 90px 1fr auto;
    gap: 10px;
    align-items: center;
    padding: 7px 0;
    border-bottom: 1px solid var(--border-color-primary);
}

.code-badge,
.page-badge,
.confidence-badge,
.finding-badge {
    display: inline-block;
    border: 1px solid var(--border-color-primary);
    border-radius: 999px;
    padding: 4px 9px;
    font-size: 12px;
    margin: 2px;
}

.similarity {
    font-size: 11px;
    opacity: 0.65;
}

.badge-row {
    margin-top: 7px;
}

.normalization-box {
    margin-top: 10px;
    font-size: 13px;
}

.evidence-box {
    margin-top: 12px;
    padding: 11px;
    border-left: 3px solid var(--border-color-primary);
}

.source-text {
    font-size: 11px;
    opacity: 0.65;
    margin-top: 4px;
}

.measurement-row {
    display: flex;
    justify-content: space-between;
    gap: 15px;
    align-items: center;
    padding: 10px 0;
    border-bottom: 1px solid var(--border-color-primary);
}

.measurement-name {
    font-weight: 600;
}

.measurement-value {
    font-size: 14px;
    margin-top: 2px;
}

.finding-container {
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
}

.empty-state {
    opacity: 0.6;
    padding: 12px 0;
}

.error-box {
    padding: 14px;
    border: 1px solid var(--border-color-primary);
    border-radius: 12px;
    font-weight: 600;
}
"""


# ============================================================
# GRADIO DASHBOARD
# ============================================================

with gr.Blocks(
    title="Clinical Document Review Assistant",
    css=CSS,
) as demo:

    gr.HTML(
        """
        <div class="hero">
            <div class="hero-title">
                Clinical Document Review Assistant
            </div>

            <div class="hero-subtitle">
                AI-assisted extraction and review of synthetic
                or de-identified clinical documents using
                BioClinicalBERT, SapBERT, RxNorm, and DailyMed.
                Results are intended for review and should not
                be treated as medical advice or definitive coding.
            </div>
        </div>
        """
    )

    with gr.Row():

        with gr.Column(scale=2):

            clinical_text = gr.Textbox(
                label="Clinical Note",
                lines=14,
                placeholder=(
                    "Paste a synthetic or de-identified "
                    "clinical note..."
                ),
            )

        with gr.Column(scale=1):

            pdf_input = gr.File(
                label="Clinical PDF",
                file_types=[".pdf"],
                type="filepath",
            )

            analyze_button = gr.Button(
                "Analyze Document",
                variant="primary",
            )

            clear_button = gr.ClearButton(
                [
                    clinical_text,
                    pdf_input,
                ]
            )

    overview_output = gr.HTML()

    with gr.Row():

        with gr.Column():

            gr.Markdown("## Conditions & ICD-10-CM Candidates")
            conditions_output = gr.HTML()

        with gr.Column():

            gr.Markdown("## Medications & RxNorm")
            medications_output = gr.HTML()

    with gr.Row():

        with gr.Column():

            gr.Markdown("## Laboratory Results")
            labs_output = gr.HTML()

        with gr.Column():

            gr.Markdown("## Vital Signs")
            vitals_output = gr.HTML()

    with gr.Row():

        with gr.Column():

            gr.Markdown("## Allergies")
            allergies_output = gr.HTML()

        with gr.Column():

            gr.Markdown("## Negated Findings")
            negated_output = gr.HTML()

    with gr.Accordion(
        "Developer / Structured Output",
        open=False,
    ):

        raw_output = gr.JSON(
            label="Structured Patient Record"
        )

    gr.Markdown(
        """
        ---
        **Model note:** ICD-10-CM results are semantic candidate
        suggestions, not definitive coding assignments. DailyMed
        content is medication-label safety evidence and does not
        represent a comprehensive patient-specific drug interaction
        assessment.
        """
    )

    analyze_button.click(
        fn=run_dashboard,
        inputs=[
            clinical_text,
            pdf_input,
        ],
        outputs=[
            overview_output,
            conditions_output,
            medications_output,
            labs_output,
            vitals_output,
            allergies_output,
            negated_output,
            raw_output,
        ],
    )


# ============================================================
# LAUNCH
# ============================================================

if __name__ == "__main__":

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
    )