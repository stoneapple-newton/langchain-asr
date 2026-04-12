"""
Multilingual medical glossary with alias and phonetic linking.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


GLOSSARY_PATH = Path(__file__).resolve().parents[1] / "data" / "medical_glossary.json"


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    normalized = normalized.lower()
    normalized = re.sub(r"[^\w\s-]+", " ", normalized, flags=re.UNICODE)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


@dataclass
class MedicalTerm:
    term_id: str
    canonical_en: str
    specialty: str
    category: str
    translations: dict[str, str] = field(default_factory=dict)
    aliases: dict[str, list[str]] = field(default_factory=dict)
    phonetic_variants: list[str] = field(default_factory=list)
    codes: dict[str, str] = field(default_factory=dict)
    notes: str = ""

    def all_surface_forms(self) -> list[str]:
        forms = [self.canonical_en, *self.translations.values(), *self.phonetic_variants]
        for values in self.aliases.values():
            forms.extend(values)
        deduped: list[str] = []
        seen: set[str] = set()
        for form in forms:
            if not form:
                continue
            key = _normalize(form)
            if key and key not in seen:
                seen.add(key)
                deduped.append(form)
        return deduped

    def to_dict(self) -> dict[str, Any]:
        return {
            "term_id": self.term_id,
            "canonical_en": self.canonical_en,
            "specialty": self.specialty,
            "category": self.category,
            "translations": self.translations,
            "aliases": self.aliases,
            "phonetic_variants": self.phonetic_variants,
            "codes": self.codes,
            "notes": self.notes,
        }


class MultilingualMedicalGlossary:
    def __init__(self, terms: list[MedicalTerm]):
        self.terms = terms

    def search(
        self,
        query: str,
        *,
        specialty: str | None = None,
        max_results: int = 5,
        min_score: float = 0.72,
    ) -> list[tuple[MedicalTerm, float, str]]:
        query_norm = _normalize(query)
        if not query_norm:
            return []
        results: list[tuple[MedicalTerm, float, str]] = []
        for term in self.terms:
            if specialty and specialty != "general" and term.specialty != specialty:
                continue
            best_score = 0.0
            best_form = ""
            for form in term.all_surface_forms():
                form_norm = _normalize(form)
                if not form_norm:
                    continue
                if form_norm in query_norm or query_norm in form_norm:
                    score = 1.0 if len(form_norm) > 3 else 0.85
                else:
                    score = SequenceMatcher(None, query_norm, form_norm).ratio()
                if score > best_score:
                    best_score = score
                    best_form = form
            if best_score >= min_score:
                results.append((term, round(best_score, 3), best_form))
        results.sort(key=lambda item: item[1], reverse=True)
        return results[:max_results]

    def detect_in_text(
        self,
        text: str,
        *,
        specialty: str | None = None,
        min_score: float = 0.82,
    ) -> list[tuple[str, MedicalTerm, float]]:
        normalized_text = _normalize(text)
        if not normalized_text:
            return []

        tokens = normalized_text.split()
        raw_tokens = text.split()
        matches: list[tuple[str, MedicalTerm, float]] = []
        seen: set[str] = set()

        for term in self.terms:
            if specialty and specialty != "general" and term.specialty != specialty:
                continue
            exact_forms = [_normalize(form) for form in term.all_surface_forms()]
            if any(form and form in normalized_text for form in exact_forms):
                for form in term.all_surface_forms():
                    form_norm = _normalize(form)
                    if form_norm and form_norm in normalized_text:
                        key = f"{term.term_id}:{form_norm}"
                        if key not in seen:
                            seen.add(key)
                            matches.append((form, term, 1.0))
                        break
                continue

            max_window = min(5, max((len(_normalize(form).split()) for form in term.all_surface_forms()), default=1))
            for window_size in range(1, min(max_window, len(tokens)) + 1):
                for start in range(0, len(tokens) - window_size + 1):
                    ngram = " ".join(tokens[start:start + window_size])
                    score_candidates = self.search(
                        ngram,
                        specialty=term.specialty if specialty and specialty != "general" else None,
                        max_results=1,
                        min_score=min_score,
                    )
                    if not score_candidates:
                        continue
                    matched_term, score, _ = score_candidates[0]
                    if matched_term.term_id != term.term_id:
                        continue
                    raw_match = " ".join(raw_tokens[start:start + window_size])
                    key = f"{term.term_id}:{_normalize(raw_match)}"
                    if key not in seen:
                        seen.add(key)
                        matches.append((raw_match, term, score))
        matches.sort(key=lambda item: (item[2], len(item[0])), reverse=True)
        return matches

    def to_dict(self) -> dict[str, Any]:
        return {"terms": [term.to_dict() for term in self.terms]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MultilingualMedicalGlossary":
        return cls(
            [
                MedicalTerm(
                    term_id=item["term_id"],
                    canonical_en=item["canonical_en"],
                    specialty=item.get("specialty", "general"),
                    category=item.get("category", "term"),
                    translations=item.get("translations", {}),
                    aliases=item.get("aliases", {}),
                    phonetic_variants=item.get("phonetic_variants", []),
                    codes=item.get("codes", {}),
                    notes=item.get("notes", ""),
                )
                for item in data.get("terms", [])
            ]
        )


def load_medical_glossary(path: str | Path | None = None) -> MultilingualMedicalGlossary:
    resolved = Path(path) if path else GLOSSARY_PATH
    raw = json.loads(resolved.read_text(encoding="utf-8"))
    return MultilingualMedicalGlossary.from_dict(raw)


def detect_medical_terms(
    text: str,
    glossary: MultilingualMedicalGlossary | None = None,
    *,
    specialty: str | None = None,
    min_score: float = 0.82,
) -> list[tuple[str, MedicalTerm, float]]:
    loaded = glossary or load_medical_glossary()
    return loaded.detect_in_text(text, specialty=specialty, min_score=min_score)


def link_medical_terms(
    text: str,
    glossary: MultilingualMedicalGlossary | None = None,
    *,
    specialty: str | None = None,
    min_score: float = 0.78,
) -> list[dict[str, Any]]:
    loaded = glossary or load_medical_glossary()
    linked: list[dict[str, Any]] = []
    seen: set[str] = set()
    for matched_text, term, score in loaded.detect_in_text(text, specialty=specialty, min_score=min_score):
        if term.term_id in seen:
            continue
        seen.add(term.term_id)
        linked.append(
            {
                "term_id": term.term_id,
                "raw_text": matched_text,
                "canonical_en": term.canonical_en,
                "specialty": term.specialty,
                "category": term.category,
                "translations": term.translations,
                "aliases": term.aliases,
                "phonetic_variants": term.phonetic_variants,
                "codes": term.codes,
                "icd_hint": term.codes.get("icd10", ""),
                "confidence": score,
            }
        )
    return linked


def build_llm_hint(
    specialty: str,
    glossary: MultilingualMedicalGlossary | None = None,
    *,
    max_terms: int = 8,
) -> str:
    loaded = glossary or load_medical_glossary()
    candidates = [term for term in loaded.terms if specialty == "general" or term.specialty == specialty]
    if not candidates:
        candidates = loaded.terms
    snippets: list[str] = []
    for term in candidates[:max_terms]:
        alias_values = []
        for values in term.aliases.values():
            alias_values.extend(values[:2])
        translations = ", ".join(
            f"{lang}:{value}" for lang, value in list(term.translations.items())[:4] if value
        )
        variants = ", ".join(term.phonetic_variants[:3])
        parts = [term.canonical_en]
        if translations:
            parts.append(f"translations[{translations}]")
        if alias_values:
            parts.append(f"aliases[{', '.join(alias_values[:3])}]")
        if variants:
            parts.append(f"asr[{variants}]")
        snippets.append(" ".join(parts))
    return " | ".join(snippets)
