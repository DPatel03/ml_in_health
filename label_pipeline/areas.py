"""Maps MeSH terms to BIO's 14 disease areas (plus 'Others'), using the term and its MeSH ancestors."""
from __future__ import annotations

from collections.abc import Iterable

# Checked in order; the first area whose markers appear among a trial's terms/ancestors wins.
# Oncology goes first so blood cancers land in Oncology, not Hematology (BIO's convention).
# Kidney disease is BIO "Others" (Renal), so it is caught before Urology.
AREA_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Oncology", ("neoplasm", "cancer", "carcinoma", "lymphoma", "leukemia", "myeloma", "melanoma", "sarcoma",
                  "glioma", "glioblastoma", "tumor", "myelodysplastic")),
    ("Infectious disease", ("infection", "virus diseases", "bacterial infections", "mycoses", "parasitic diseases",
                            "hiv", "hepatitis", "influenza", "tuberculosis", "malaria", "sepsis", "covid")),
    ("Autoimmune", ("autoimmune diseases", "arthritis, rheumatoid", "psoriasis", "lupus", "inflammatory bowel",
                    "crohn disease", "colitis, ulcerative", "multiple sclerosis", "spondylitis", "sjogren")),
    ("Allergy", ("hypersensitivity", "rhinitis, allergic", "urticaria", "anaphylaxis")),
    ("Hematology", ("hematologic diseases", "anemia", "hemophilia", "thrombocytopenia", "blood coagulation disorders",
                    "neutropenia", "hemoglobinopathies")),
    ("Ophthalmology", ("eye diseases", "retinal", "macular", "glaucoma", "uveitis")),
    ("Psychiatry", ("mental disorders", "depressive", "schizophrenia", "bipolar", "anxiety", "substance-related",
                    "attention deficit")),
    ("Neurology", ("nervous system diseases", "alzheimer", "parkinson", "epilepsy", "migraine", "neuralgia", "pain")),
    ("Cardiovascular", ("cardiovascular diseases", "heart", "hypertension", "vascular diseases", "atrial fibrillation",
                        "thrombosis", "stroke")),
    ("Endocrine", ("diabetes", "endocrine system diseases", "thyroid", "growth disorders", "hypogonadism")),
    ("Metabolic", ("metabolic diseases", "obesity", "dyslipidemias", "hypercholesterolemia", "lipid metabolism")),
    ("Respiratory", ("respiratory tract diseases", "asthma", "pulmonary", "lung diseases", "cystic fibrosis")),
    ("Gastroenterology", ("digestive system diseases", "gastro", "liver diseases", "irritable bowel")),
    ("Others", ("kidney diseases", "renal insufficiency")),
    ("Urology", ("urologic diseases", "urinary", "prostatic hyperplasia", "erectile")),
)


def area_for(terms: Iterable[str]) -> str:
    text = " | ".join(t.lower() for t in terms)
    if not text:
        return "Unknown"
    for area, markers in AREA_RULES:
        if any(m in text for m in markers):
            return area
    return "Others"
