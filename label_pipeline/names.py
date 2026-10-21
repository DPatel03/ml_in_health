"""Drug-name cleaning shared by the ChEMBL, Drugs@FDA and AACT matchers."""
from __future__ import annotations

import re

SALTS = (
    "bisulfate", "ditosylate", "besilate", "mesilate", "hydrochloride", "dihydrochloride", "hcl", "hydrobromide",
    "sodium", "disodium", "potassium", "calcium", "mesylate", "dimesylate", "maleate", "dimaleate", "citrate",
    "sulfate", "acetate", "tartrate", "bitartrate", "besylate", "fumarate", "succinate", "phosphate", "bromide",
    "monohydrate", "dihydrate", "trihydrate", "hyclate", "tosylate", "malate", "lactate", "hemihydrate",
    "anhydrous", "free base", "camsylate", "meglumine", "tromethamine", "hemifumarate", "esylate",
)
_SALT_TAIL = re.compile(rf"(\s+({'|'.join(SALTS)}))+\s*$", re.I)
_FORMS = re.compile(
    r"\b(tablets?|capsules?|injection|injectable|oral|iv|intravenous|subcutaneous|sc|topical|cream|gel|ointment|"
    r"solution|infusion|suspension|inhaler|inhalation|nasal spray|spray|patch|eye drops|drops|extended[- ]release|"
    r"delayed[- ]release|er|xr|sr|cr|odt|film[- ]coated|for injection|powder|lyophilized|concentrate|syrup|"
    r"suppository|vaginal|ophthalmic)\b", re.I)
_DOSE = re.compile(r"\b\d+([.,]\d+)?\s*(mg/kg|mg/m2|mg/m²|mcg/kg|mg/ml|mg|mcg|µg|ug|g|ml|iu|units?|u|%|mmol|meq|nmol)\b", re.I)
_LONE_NUMBER = re.compile(r"(?<![a-z0-9-])\d+([.,]\d+)?(?![a-z0-9-])", re.I)
_PREFIX = re.compile(r"^(n|l|d|dl|\[1[14]c\]|1[14]c)[- ]", re.I)  # n-acetylcysteine, l-arginine, [14C]-labelled drug
_BIOSIMILAR_SUFFIX = re.compile(r"-[a-z]{4}$")  # rituximab-abbs
_COMBINATION = re.compile(r"\+| and | with |/|,|;|\bplus\b|regimen|\bfolfox|\bfolfiri|\br-?chop\b|\bchop\b", re.I)
_NOT_A_DRUG = re.compile(
    r"placebo|saline|vehicle|sham|sugar pill|standard of care|best supportive care|usual care|chemotherapy$|"
    r"^control$|^drug$|^study drug$|investigational product", re.I)
_RESEARCH_CODE = re.compile(r"^[a-z]{1,6}\d{3,}[a-z0-9]*$")


def compact(name: str) -> str:
    """Lowercase alphanumerics only: 'MK-3475' -> 'mk3475', 'Naltrexone HCl' -> 'naltrexonehcl'."""
    s = _PREFIX.sub("", str(name).strip().lower())
    return re.sub(r"[^a-z0-9]", "", s)


def strip_salt(name: str) -> str:
    base = _SALT_TAIL.sub("", str(name).strip())
    return base if len(compact(base)) >= 3 else str(name)


def strip_biosimilar_suffix(name: str) -> str:
    return _BIOSIMILAR_SUFFIX.sub("", str(name).strip().lower())


def is_combination(name: str) -> bool:
    return bool(_COMBINATION.search(str(name)))


def is_not_a_drug(name: str) -> bool:
    return bool(_NOT_A_DRUG.search(str(name).strip()))


def is_research_code(key: str) -> bool:
    return bool(_RESEARCH_CODE.match(key))


def name_keys(name: str, min_len: int = 3) -> set[str]:
    """All lookup keys for one free-text drug name: with/without parentheses, dose, form and salt words."""
    s = str(name).lower().replace("®", "").replace("™", "")
    variants = [re.sub(r"\(.*?\)|\[.*?\]", " ", s), *re.findall(r"\((.*?)\)", s), s]
    keys: set[str] = set()
    for v in variants:
        v = _FORMS.sub(" ", _DOSE.sub(" ", v))
        v = _LONE_NUMBER.sub(" ", v)
        v = re.sub(r"\s+", " ", v).strip(" -,;:/")
        for form in {v, strip_salt(v)}:
            if len(k := compact(form)) >= min_len:
                keys.add(k)
    return keys
