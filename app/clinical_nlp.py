import re

from app.references import DRUG_LIST, DDI_REF

DOSE_UNIT = r'(?:mg|g|mcg|ml|l|units|IU|mEq|meq|pills|tablets|capsules)'

ROUTE_PAT = r'(?:PO|IV|IM|SC|SL|PR|topical|inhalation|nasal|ophthalmic|otic)'

FREQ_PAT = r'(?:QD|BID|TID|QID|PRN|daily|weekly|monthly|every\s+\d+\s+(?:hours?|days?|weeks?|months?))'

DRUG_PAT = (r'\b(' + '|'.join(re.escape(drug) for drug in DRUG_LIST)
            + r')\b')

MED_PAT = (DRUG_PAT + r'(?:[\s,]+(\d+(?:\.\d+)?)\s*(' + DOSE_UNIT + r'))?'
    + r'(?:[\s,]+(' + ROUTE_PAT + r'))?'
    + r'(?:[\s,]+(' + FREQ_PAT + r'))?'
)


def extract_meds(text):
    seen = set()
    meds = []
    for match in re.finditer(MED_PAT,text, re.IGNORECASE):
        drug = match.group(1).lower()

        if drug not in seen:
            seen.add(drug)

            meds.append({
                "drug": drug,
                "dose": (match.group(2) or "").strip() ,
                "unit": (match.group(3) or "").strip(),
                "route": (match.group(4) or "").upper(),
                "frequency": (match.group(5) or "").upper()
            })

    return meds


def check_ddi(medication_list):
    med_names = {med["drug"].lower() for med in medication_list}

    flagged = []

    for drug1, drug2, severity, message in DDI_REF:
        if drug1 in med_names and drug2 in med_names:
            flagged.append({
                "drug1": drug1,
                "drug2": drug2,
                "severity": severity,
                "message": message
            })

    return flagged


def analyze_note(text):
    if not text or not text.strip():
        return {
            "medications": [],
            "ddis": []
        }

    meds = extract_meds(text)

    interactions = check_ddi(meds)

    return {
        "medications": meds,
        "drug_interactions": interactions
    }