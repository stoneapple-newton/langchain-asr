"""
Multilingual Medical Glossary
=============================
Medical terminology support across multiple languages to assist LLM
in identifying and verifying medical terms in transcriptions.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class MedicalTerm:
    """A medical term with multilingual support."""
    term_id: str  # Unique identifier
    english: str  # Primary English term
    category: str  # e.g., "anatomy", "disease", "procedure", "medication"
    translations: dict[str, str] = field(default_factory=dict)  # lang_code -> term
    synonyms: list[str] = field(default_factory=list)  # Alternative names in English
    definitions: dict[str, str] = field(default_factory=dict)  # lang_code -> definition
    abbreviations: list[str] = field(default_factory=list)  # Common abbreviations
    related_terms: list[str] = field(default_factory=list)  # Related term_ids
    
    def get_term(self, language: str = "en") -> str:
        """Get term in specified language."""
        if language == "en" or language not in self.translations:
            return self.english
        return self.translations[language]
    
    def get_all_forms(self) -> list[str]:
        """Get all forms of this term (English, translations, synonyms)."""
        forms = [self.english, *self.translations.values(), *self.synonyms]
        return [f for f in forms if f]
    
    def matches(self, text: str, case_sensitive: bool = False) -> bool:
        """Check if text matches this term or any of its forms."""
        if not case_sensitive:
            text = text.lower()
            forms = [f.lower() for f in self.get_all_forms()]
        else:
            forms = self.get_all_forms()
        
        return text in forms or any(f in text for f in forms)


class MultilingualMedicalGlossary:
    """A multilingual glossary of medical terms."""
    
    def __init__(self):
        self.terms: dict[str, MedicalTerm] = {}  # term_id -> MedicalTerm
        self._by_category: dict[str, list[str]] = {}  # category -> term_ids
        self._by_language: dict[str, list[str]] = {}  # lang -> term_ids
        self._search_index: dict[str, set[str]] = {}  # word -> term_ids
    
    def add_term(self, term: MedicalTerm) -> None:
        """Add a medical term to the glossary."""
        self.terms[term.term_id] = term
        
        # Index by category
        if term.category not in self._by_category:
            self._by_category[term.category] = []
        self._by_category[term.category].append(term.term_id)
        
        # Index by language
        for lang in term.translations.keys():
            if lang not in self._by_language:
                self._by_language[lang] = []
            self._by_language[lang].append(term.term_id)
        
        # Build search index
        for form in term.get_all_forms():
            words = form.lower().split()
            for word in words:
                if word not in self._search_index:
                    self._search_index[word] = set()
                self._search_index[word].add(term.term_id)
    
    def get_term(self, term_id: str) -> MedicalTerm | None:
        """Get a term by ID."""
        return self.terms.get(term_id)
    
    def find_by_category(self, category: str) -> list[MedicalTerm]:
        """Find all terms in a category."""
        term_ids = self._by_category.get(category, [])
        return [self.terms[tid] for tid in term_ids]
    
    def search(self, query: str, max_results: int = 10) -> list[MedicalTerm]:
        """Search for terms matching the query."""
        query_lower = query.lower()
        query_words = query_lower.split()
        
        # Find matching term IDs
        matching_ids: dict[str, int] = {}
        for word in query_words:
            for term_id in self._search_index.get(word, set()):
                matching_ids[term_id] = matching_ids.get(term_id, 0) + 1
        
        # Score and sort
        scored = [(tid, score) for tid, score in matching_ids.items()]
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return [self.terms[tid] for tid, _ in scored[:max_results]]
    
    def detect_in_text(self, text: str) -> list[tuple[str, MedicalTerm]]:
        """Detect medical terms in text. Returns list of (matched_text, term)."""
        results = []
        text_lower = text.lower()
        
        for term in self.terms.values():
            for form in term.get_all_forms():
                if form.lower() in text_lower:
                    # Find all occurrences
                    for match in re.finditer(re.escape(form.lower()), text_lower):
                        start, end = match.span()
                        matched_text = text[start:end]
                        results.append((matched_text, term))
        
        # Remove duplicates (keep longest matches)
        results = sorted(results, key=lambda x: len(x[0]), reverse=True)
        filtered = []
        covered = set()
        for matched_text, term in results:
            # Check if this is covered by a longer match
            if matched_text.lower() not in covered:
                filtered.append((matched_text, term))
                covered.add(matched_text.lower())
        
        return filtered
    
    def to_dict(self) -> dict[str, Any]:
        """Convert glossary to dictionary."""
        return {
            "terms": [
                {
                    "term_id": t.term_id,
                    "english": t.english,
                    "category": t.category,
                    "translations": t.translations,
                    "synonyms": t.synonyms,
                    "definitions": t.definitions,
                    "abbreviations": t.abbreviations,
                    "related_terms": t.related_terms,
                }
                for t in self.terms.values()
            ]
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MultilingualMedicalGlossary":
        """Create glossary from dictionary."""
        glossary = cls()
        for term_data in data.get("terms", []):
            term = MedicalTerm(
                term_id=term_data["term_id"],
                english=term_data["english"],
                category=term_data["category"],
                translations=term_data.get("translations", {}),
                synonyms=term_data.get("synonyms", []),
                definitions=term_data.get("definitions", {}),
                abbreviations=term_data.get("abbreviations", []),
                related_terms=term_data.get("related_terms", []),
            )
            glossary.add_term(term)
        return glossary
    
    def save(self, path: str | Path) -> None:
        """Save glossary to JSON file."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    
    @classmethod
    def load(cls, path: str | Path) -> "MultilingualMedicalGlossary":
        """Load glossary from JSON file."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)


def create_default_glossary() -> MultilingualMedicalGlossary:
    """Create a default glossary with common medical terms."""
    glossary = MultilingualMedicalGlossary()
    
    # Anatomy terms
    anatomy_terms = [
        MedicalTerm(
            term_id="anatomy_heart",
            english="heart",
            category="anatomy",
            translations={
                "es": "corazón",
                "fr": "cœur",
                "de": "Herz",
                "it": "cuore",
                "pt": "coração",
                "zh": "心脏",
                "ja": "心臓",
                "ar": "قلب",
                "hi": "हृदय",
            },
            synonyms=["cardiac", "myocardium"],
            abbreviations=["H"],
        ),
        MedicalTerm(
            term_id="anatomy_lung",
            english="lung",
            category="anatomy",
            translations={
                "es": "pulmón",
                "fr": "poumon",
                "de": "Lunge",
                "it": "polmone",
                "pt": "pulmão",
                "zh": "肺",
                "ja": "肺",
                "ar": "رئة",
                "hi": "फेफड़ा",
            },
            synonyms=["pulmonary"],
        ),
        MedicalTerm(
            term_id="anatomy_brain",
            english="brain",
            category="anatomy",
            translations={
                "es": "cerebro",
                "fr": "cerveau",
                "de": "Gehirn",
                "it": "cervello",
                "pt": "cérebro",
                "zh": "大脑",
                "ja": "脳",
                "ar": "دماغ",
                "hi": "मस्तिष्क",
            },
            synonyms=["cerebral", "encephalon"],
        ),
        MedicalTerm(
            term_id="anatomy_liver",
            english="liver",
            category="anatomy",
            translations={
                "es": "hígado",
                "fr": "foie",
                "de": "Leber",
                "it": "fegato",
                "pt": "fígado",
                "zh": "肝脏",
                "ja": "肝臓",
                "ar": "كبد",
                "hi": "यकृत",
            },
            synonyms=["hepatic"],
        ),
        MedicalTerm(
            term_id="anatomy_kidney",
            english="kidney",
            category="anatomy",
            translations={
                "es": "riñón",
                "fr": "rein",
                "de": "Niere",
                "it": "rene",
                "pt": "rim",
                "zh": "肾脏",
                "ja": "腎臓",
                "ar": "كلية",
                "hi": "गुर्दा",
            },
            synonyms=["renal"],
        ),
    ]
    
    # Disease terms
    disease_terms = [
        MedicalTerm(
            term_id="disease_diabetes",
            english="diabetes",
            category="disease",
            translations={
                "es": "diabetes",
                "fr": "diabète",
                "de": "Diabetes",
                "it": "diabete",
                "pt": "diabetes",
                "zh": "糖尿病",
                "ja": "糖尿病",
                "ar": "السكري",
                "hi": "मधुमेह",
            },
            synonyms=["diabetes mellitus", "sugar disease"],
            abbreviations=["DM"],
        ),
        MedicalTerm(
            term_id="disease_hypertension",
            english="hypertension",
            category="disease",
            translations={
                "es": "hipertensión",
                "fr": "hypertension",
                "de": "Bluthochdruck",
                "it": "ipertensione",
                "pt": "hipertensão",
                "zh": "高血压",
                "ja": "高血圧症",
                "ar": "ارتفاع ضغط الدم",
                "hi": "उच्च रक्तचाप",
            },
            synonyms=["high blood pressure"],
            abbreviations=["HTN", "HT", "HBP"],
        ),
        MedicalTerm(
            term_id="disease_pneumonia",
            english="pneumonia",
            category="disease",
            translations={
                "es": "neumonía",
                "fr": "pneumonie",
                "de": "Lungenentzündung",
                "it": "polmonite",
                "pt": "pneumonia",
                "zh": "肺炎",
                "ja": "肺炎",
                "ar": "التهاب رئوي",
                "hi": "निमोनिया",
            },
        ),
        MedicalTerm(
            term_id="disease_arrhythmia",
            english="arrhythmia",
            category="disease",
            translations={
                "es": "arritmia",
                "fr": "arythmie",
                "de": "Herzrhythmusstörung",
                "it": "aritmia",
                "pt": "arritmia",
                "zh": "心律失常",
                "ja": "不整脈",
                "ar": "عدم انتظام ضربات القلب",
                "hi": "अनियमित धड़कन",
            },
            synonyms=["irregular heartbeat", "dysrhythmia"],
        ),
    ]
    
    # Procedure terms
    procedure_terms = [
        MedicalTerm(
            term_id="procedure_surgery",
            english="surgery",
            category="procedure",
            translations={
                "es": "cirugía",
                "fr": "chirurgie",
                "de": "Operation",
                "it": "chirurgia",
                "pt": "cirurgia",
                "zh": "手术",
                "ja": "手術",
                "ar": "جراحة",
                "hi": "शल्य चिकित्सा",
            },
            synonyms=["operation", "procedure"],
            abbreviations=["Sx", "OP"],
        ),
        MedicalTerm(
            term_id="procedure_biopsy",
            english="biopsy",
            category="procedure",
            translations={
                "es": "biopsia",
                "fr": "biopsie",
                "de": "Biopsie",
                "it": "biopsia",
                "pt": "biópsia",
                "zh": "活检",
                "ja": "生検",
                "ar": "خزعة",
                "hi": "जीवाणु परीक्षण",
            },
        ),
        MedicalTerm(
            term_id="procedure_mri",
            english="magnetic resonance imaging",
            category="procedure",
            translations={
                "es": "resonancia magnética",
                "fr": "imagerie par résonance magnétique",
                "de": "Magnetresonanztomographie",
                "it": "risonanza magnetica",
                "pt": "ressonância magnética",
                "zh": "磁共振成像",
                "ja": "磁気共鳴画像法",
                "ar": "التصوير بالرنين المغناطيسي",
                "hi": "चुंबकीय अनु resonancia इमेजिंग",
            },
            synonyms=["MRI scan", "MR imaging"],
            abbreviations=["MRI"],
        ),
        MedicalTerm(
            term_id="procedure_ct",
            english="computed tomography",
            category="procedure",
            translations={
                "es": "tomografía computarizada",
                "fr": "tomodensitométrie",
                "de": "Computertomographie",
                "it": "tomografia computerizzata",
                "pt": "tomografia computadorizada",
                "zh": "计算机断层扫描",
                "ja": "コンピュータ断層撮影",
                "ar": "التصوير المقطعي المحوسب",
                "hi": "कंप्यूटेड टोमोग्राफी",
            },
            synonyms=["CT scan", "CAT scan"],
            abbreviations=["CT"],
        ),
    ]
    
    # Medication terms
    medication_terms = [
        MedicalTerm(
            term_id="medication_aspirin",
            english="aspirin",
            category="medication",
            translations={
                "es": "aspirina",
                "fr": "aspirine",
                "de": "Aspirin",
                "it": "aspirina",
                "pt": "aspirina",
                "zh": "阿司匹林",
                "ja": "アスピリン",
                "ar": "أسبرين",
                "hi": "एस्पिरिन",
            },
            synonyms=["acetylsalicylic acid"],
            abbreviations=["ASA"],
        ),
        MedicalTerm(
            term_id="medication_antibiotic",
            english="antibiotic",
            category="medication",
            translations={
                "es": "antibiótico",
                "fr": "antibiotique",
                "de": "Antibiotikum",
                "it": "antibiotico",
                "pt": "antibiótico",
                "zh": "抗生素",
                "ja": "抗生物質",
                "ar": "مضاد حيوي",
                "hi": "प्रतिजैविक",
            },
        ),
        MedicalTerm(
            term_id="medication_insulin",
            english="insulin",
            category="medication",
            translations={
                "es": "insulina",
                "fr": "insuline",
                "de": "Insulin",
                "it": "insulina",
                "pt": "insulina",
                "zh": "胰岛素",
                "ja": "インスリン",
                "ar": "الأنسولين",
                "hi": "इंसुलिन",
            },
        ),
    ]
    
    # Vital signs
    vital_terms = [
        MedicalTerm(
            term_id="vital_bp",
            english="blood pressure",
            category="vital_signs",
            translations={
                "es": "presión arterial",
                "fr": "tension artérielle",
                "de": "Blutdruck",
                "it": "pressione sanguigna",
                "pt": "pressão arterial",
                "zh": "血压",
                "ja": "血圧",
                "ar": "ضغط الدم",
                "hi": "रक्तचाप",
            },
            abbreviations=["BP"],
        ),
        MedicalTerm(
            term_id="vital_hr",
            english="heart rate",
            category="vital_signs",
            translations={
                "es": "frecuencia cardíaca",
                "fr": "fréquence cardiaque",
                "de": "Herzfrequenz",
                "it": "frequenza cardiaca",
                "pt": "frequência cardíaca",
                "zh": "心率",
                "ja": "心拍数",
                "ar": "معدل ضربات القلب",
                "hi": "हृदय गति",
            },
            abbreviations=["HR", "pulse"],
        ),
        MedicalTerm(
            term_id="vital_o2sat",
            english="oxygen saturation",
            category="vital_signs",
            translations={
                "es": "saturación de oxígeno",
                "fr": "saturation en oxygène",
                "de": "Sauerstoffsättigung",
                "it": "saturazione di ossigeno",
                "pt": "saturação de oxigênio",
                "zh": "血氧饱和度",
                "ja": "酸素飽和度",
                "ar": "تشبع الأكسجين",
                "hi": "ऑक्सीजन संतृप्ति",
            },
            abbreviations=["SpO2", "O2 sat"],
        ),
    ]
    
    # Add all terms
    for term in anatomy_terms + disease_terms + procedure_terms + medication_terms + vital_terms:
        glossary.add_term(term)
    
    return glossary


def load_medical_glossary(path: str | Path | None = None) -> MultilingualMedicalGlossary:
    """Load medical glossary from file or create default."""
    if path is not None and Path(path).exists():
        return MultilingualMedicalGlossary.load(path)
    return create_default_glossary()


def detect_medical_terms(
    text: str,
    glossary: MultilingualMedicalGlossary | None = None,
) -> list[tuple[str, MedicalTerm]]:
    """Detect medical terms in text."""
    if glossary is None:
        glossary = create_default_glossary()
    return glossary.detect_in_text(text)
