"""ChEMBL drug dictionary: every molecule that reached clinical trials, with its names and research codes.

Only names and the parent-compound link are kept. ChEMBL's own max_phase / first_approval fields are deliberately
not stored: they encode the outcome we are trying to predict.
"""
from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import httpx

from .names import compact, name_keys, strip_salt

log = logging.getLogger(__name__)

API = "https://www.ebi.ac.uk/chembl/api/data/molecule.json"
FIELDS = "molecule_chembl_id,pref_name,molecule_type,molecule_synonyms,molecule_hierarchy"
PAGE_SIZE = 1000


async def _get_page(client: httpx.AsyncClient, offset: int, sem: asyncio.Semaphore, retries: int) -> list[dict[str, Any]]:
    params = {"max_phase__gte": 0.5, "limit": PAGE_SIZE, "offset": offset, "only": FIELDS}
    for attempt in range(1, retries + 1):
        try:
            async with sem:
                r = await client.get(API, params=params)
            r.raise_for_status()
            return r.json()["molecules"]
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            if attempt == retries:
                raise RuntimeError(f"ChEMBL page at offset {offset} failed after {retries} tries: {exc}") from exc
            wait = 2 ** attempt
            log.warning("ChEMBL offset %s failed (%s); retrying in %ss", offset, exc, wait)
            await asyncio.sleep(wait)
    return []  # unreachable


async def fetch_clinical_molecules(out: Path, concurrency: int = 3, retries: int = 4, timeout: float = 120.0) -> int:
    """Download all clinical-stage molecules to a JSON-lines file. Returns the number of molecules written."""
    async with httpx.AsyncClient(timeout=timeout, headers={"Accept": "application/json"}) as client:
        first = await client.get(API, params={"max_phase__gte": 0.5, "limit": 1, "only": "molecule_chembl_id"})
        first.raise_for_status()
        total = int(first.json()["page_meta"]["total_count"])
        sem = asyncio.Semaphore(concurrency)
        pages = await asyncio.gather(*(_get_page(client, off, sem, retries) for off in range(0, total, PAGE_SIZE)))
    molecules = [m for page in pages for m in page]
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    with tmp.open("w") as fh:
        for m in molecules:
            fh.write(json.dumps(m) + "\n")
    tmp.replace(out)
    if len(molecules) != total:
        log.warning("ChEMBL reported %s molecules but %s were downloaded", total, len(molecules))
    return len(molecules)


def load_molecules(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"ChEMBL dictionary not found: {path} (run with --fetch-chembl)")
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _names(m: dict[str, Any]) -> Iterable[str]:
    if m.get("pref_name"):
        yield m["pref_name"]
    for s in m.get("molecule_synonyms") or []:
        for field in ("molecule_synonym", "synonyms"):
            if s.get(field):
                yield s[field]


def build_synonym_index(molecules: list[dict[str, Any]]) -> tuple[dict[str, str], dict[str, str]]:
    """Map every name key to one parent ChEMBL ID, plus a parent -> display name map.

    Keys that point at more than one parent compound are dropped as ambiguous rather than guessed.
    """
    owners: dict[str, set[str]] = defaultdict(set)
    display: dict[str, str] = {}
    for m in molecules:
        hierarchy = m.get("molecule_hierarchy") or {}
        parent = hierarchy.get("parent_chembl_id") or m["molecule_chembl_id"]
        if m.get("pref_name") and (parent == m["molecule_chembl_id"] or parent not in display):
            display[parent] = m["pref_name"].lower()
        for name in _names(m):
            for key in name_keys(name) | {compact(strip_salt(name))}:
                if len(key) >= 3:
                    owners[key].add(parent)
    index = {k: next(iter(v)) for k, v in owners.items() if len(v) == 1}
    log.info("ChEMBL index: %s keys (%s ambiguous keys dropped)", len(index), len(owners) - len(index))
    return index, display
