DRUG_LIST = sorted([
    "metoprolol",
    "lisinopril",
    "furosemide",
    "warfarin",
    "aspirin",
    "spironolactone",
    "rivaroxaban",
    "metformin",
    "insulin",
    "vancomycin",
    "fluconazole",
    "amiodarone",
    "digoxin",
], key=len, reverse=True)


DDI_REF = [
    (
        "warfarin",
        "aspirin",
        "HIGH",
        "Increased bleeding risk"
    ),
    (
        "warfarin",
        "fluconazole",
        "HIGH",
        "Potential increased warfarin effect"
    ),
    (
        "digoxin",
        "amiodarone",
        "HIGH",
        "Potential increased digoxin exposure"
    ),
    (
        "lisinopril",
        "spironolactone",
        "MOD",
        "Potential hyperkalemia risk"
    ),
]