"""
TermContextStore — persistent Chroma-backed store for medical terms.

Each term is stored as a LangChain Document whose ``page_content`` is a
pipe-separated string of all surface forms (English canonical + translations
+ phonetic variants). This lets the embeddings capture cross-lingual
similarity so that querying "cardiology" in any language retrieves the
same term.

Metadata fields
---------------
term_id         : unique key, e.g. "cardio_afib" or "extracted_<hash>"
canonical_en    : English canonical form, e.g. "atrial fibrillation"
specialty       : medical specialty, e.g. "cardiology"
category        : "condition" | "procedure" | "medication" | "anatomy" | …
icd_code        : ICD-10 prefix when known, e.g. "I48"
language_coverage : comma-joined BCP-47 codes present in translations
phonetic_variants : pipe-joined list of ASR mishearing variants
frequency       : int — how many sessions have seen this term
last_session    : ISO-date string of the most recent session
source          : "glossary" | "extracted"

Usage
-----
    from deep_research.medical_asr_context_system.context_store.store import TermContextStore

    store = TermContextStore()              # uses default persist dir
    store.bootstrap_from_glossary()        # seed on first run
    results = store.query("cardiology", "en", top_k=20)
    store.upsert_term({...})
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from langchain_core.documents import Document

from config import create_embeddings

_DEFAULT_STORE_DIR = REPO_ROOT / "data" / "medical_context_store"

# Lazy import to avoid hard dependency at module level
_chroma_mod = None


def _get_chroma():
    global _chroma_mod
    if _chroma_mod is None:
        from langchain_chroma import Chroma  # noqa: PLC0415
        _chroma_mod = Chroma
    return _chroma_mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_page_content(term: dict[str, Any]) -> str:
    """Produce the embeddable surface-form string for a term dict."""
    parts: list[str] = []
    if term.get("canonical_en"):
        parts.append(term["canonical_en"])
    for lang_key in ("es", "fr", "de", "pt", "ar", "zh_pinyin", "ja_romaji",
                     "zh", "ja", "hi"):
        val = term.get(lang_key, "")
        if val and val not in parts:
            parts.append(val)
    # Include phonetic variants so ASR mishearings are embedded too
    for variant in _parse_pipe(term.get("phonetic_variants", "")):
        if variant and variant not in parts:
            parts.append(variant)
    return " | ".join(parts)


def _parse_pipe(value: str | list | None) -> list[str]:
    """Accept a pipe-separated string or a list; return a list of strings."""
    if not value:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if v]
    return [v.strip() for v in str(value).split("|") if v.strip()]


def _stable_id(canonical_en: str) -> str:
    """Deterministic term_id derived from the canonical English form."""
    slug = canonical_en.lower().replace(" ", "_")
    digest = hashlib.md5(canonical_en.lower().encode()).hexdigest()[:6]
    return f"term_{slug}_{digest}"


def _glossary_term_to_dict(term: dict[str, Any]) -> dict[str, Any]:
    """Convert a raw glossary JSON term to the store's normalised schema."""
    canonical = term.get("en") or term.get("canonical_en", "")
    phonetic = term.get("phonetic_variants", [])
    langs = [k for k in ("es", "fr", "de", "pt", "ar", "zh_pinyin", "ja_romaji")
             if term.get(k)]
    return {
        "term_id": _stable_id(canonical),
        "canonical_en": canonical,
        "specialty": term.get("specialty", "general"),
        "category": term.get("category", "general"),
        "icd_code": term.get("icd_hint", ""),
        "language_coverage": ",".join(langs),
        "phonetic_variants": "|".join(phonetic) if isinstance(phonetic, list) else phonetic,
        "frequency": 1,
        "last_session": "",
        "source": "glossary",
        # Keep translations so _build_page_content can use them
        "es": term.get("es", ""),
        "fr": term.get("fr", ""),
        "de": term.get("de", ""),
        "pt": term.get("pt", ""),
        "ar": term.get("ar", ""),
        "zh_pinyin": term.get("zh_pinyin", ""),
        "ja_romaji": term.get("ja_romaji", ""),
    }


# ---------------------------------------------------------------------------
# TermContextStore
# ---------------------------------------------------------------------------

class TermContextStore:
    """
    Chroma-backed persistent store for multilingual medical terms.

    Parameters
    ----------
    persist_dir : path-like, optional
        Directory where Chroma writes its SQLite + vector index files.
        Defaults to ``<repo_root>/data/medical_context_store``.
    embeddings_profile : str
        Config profile name passed to ``create_embeddings()``.
        Defaults to ``"default"`` (nomic-embed-text via Ollama).
        Switch to ``"medical"`` when OpenAI embeddings are configured.
    collection_name : str
        Chroma collection name. Change to namespace separate stores.
    """

    def __init__(
        self,
        persist_dir: str | Path | None = None,
        embeddings_profile: str = "default",
        collection_name: str = "medical_terms",
    ) -> None:
        self._persist_dir = Path(persist_dir or _DEFAULT_STORE_DIR)
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._embeddings = create_embeddings(embeddings_profile)
        self._collection_name = collection_name
        self._store: Any = None  # lazy-initialised

    # ------------------------------------------------------------------
    # Lazy store access
    # ------------------------------------------------------------------

    def _get_store(self) -> Any:
        if self._store is None:
            Chroma = _get_chroma()
            self._store = Chroma(
                collection_name=self._collection_name,
                persist_directory=str(self._persist_dir),
                embedding_function=self._embeddings,
            )
        return self._store

    # ------------------------------------------------------------------
    # Bootstrap
    # ------------------------------------------------------------------

    def bootstrap_from_glossary(
        self,
        glossary_path: str | Path | None = None,
        force: bool = False,
    ) -> int:
        """
        Seed the store from the project's medical glossary JSON.

        Reads ``deep_research/transcription_correction_agent/data/medical_glossary.json``
        by default (the glossary with the richest multilingual coverage).

        Parameters
        ----------
        glossary_path : override path to a glossary JSON file
        force         : re-insert even if the collection already has documents

        Returns
        -------
        int : number of terms inserted (0 if already seeded and force=False)
        """
        store = self._get_store()
        if not force and store._collection.count() > 0:
            return 0

        if glossary_path is None:
            glossary_path = (
                REPO_ROOT
                / "deep_research"
                / "transcription_correction_agent"
                / "data"
                / "medical_glossary.json"
            )

        raw = json.loads(Path(glossary_path).read_text(encoding="utf-8"))
        terms = raw.get("terms", [])

        docs: list[Document] = []
        for term in terms:
            normalised = _glossary_term_to_dict(term)
            metadata = {k: v for k, v in normalised.items()
                        if k not in ("es", "fr", "de", "pt", "ar", "zh_pinyin", "ja_romaji")}
            docs.append(Document(
                page_content=_build_page_content(normalised),
                metadata=metadata,
                id=normalised["term_id"],
            ))

        if docs:
            store.add_documents(docs)

        return len(docs)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        specialty: str,
        language: str,
        top_k: int = 30,
        extra_query: str = "",
    ) -> list[dict[str, Any]]:
        """
        Retrieve the most relevant medical terms for a specialty/language.

        The query string combines the specialty name, the language code, and
        any additional free-text hint (e.g. a topic detected from the audio
        file name or patient notes).

        Returns a list of metadata dicts (one per matched term) sorted by
        relevance.
        """
        store = self._get_store()
        if store._collection.count() == 0:
            return []

        query_str = f"{specialty} medical terms {language}"
        if extra_query:
            query_str = f"{query_str} {extra_query}"

        try:
            results = store.similarity_search_with_score(query_str, k=top_k)
        except Exception:
            return []

        items: list[dict[str, Any]] = []
        for doc, score in results:
            entry = dict(doc.metadata)
            entry["_score"] = float(score)
            entry["_page_content"] = doc.page_content
            items.append(entry)

        # Secondary sort: prefer specialty match, then frequency
        items.sort(
            key=lambda x: (
                0 if x.get("specialty") == specialty else 1,
                -x.get("frequency", 1),
                x["_score"],
            )
        )
        return items

    # ------------------------------------------------------------------
    # Upsert
    # ------------------------------------------------------------------

    def upsert_term(
        self,
        term: dict[str, Any],
        session_id: str = "",
    ) -> str:
        """
        Insert a new term or increment the frequency of an existing one.

        Parameters
        ----------
        term       : dict conforming to the store's schema (must have ``canonical_en``)
        session_id : session identifier for provenance

        Returns
        -------
        str : "added" | "updated" | "skipped"
        """
        canonical = (term.get("canonical_en") or "").strip()
        if not canonical:
            return "skipped"

        store = self._get_store()
        term_id = term.get("term_id") or _stable_id(canonical)

        # Check if this term already exists by term_id
        try:
            existing = store._collection.get(ids=[term_id])
            if existing and existing.get("ids"):
                # Term exists — increment frequency and update last_session
                old_meta = existing["metadatas"][0]
                new_freq = int(old_meta.get("frequency", 1)) + 1
                new_meta = {**old_meta, "frequency": new_freq, "last_session": session_id}
                store._collection.update(
                    ids=[term_id],
                    metadatas=[new_meta],
                )
                return "updated"
        except Exception:
            pass

        # New term — build the full document and insert
        normalised = {**term, "term_id": term_id, "source": term.get("source", "extracted")}
        if session_id:
            normalised["last_session"] = session_id
        normalised.setdefault("frequency", 1)

        metadata = {
            k: v for k, v in normalised.items()
            if k not in ("es", "fr", "de", "pt", "ar", "zh_pinyin", "ja_romaji")
            and isinstance(v, (str, int, float, bool))
        }

        doc = Document(
            page_content=_build_page_content(normalised),
            metadata=metadata,
            id=term_id,
        )
        try:
            store.add_documents([doc])
            return "added"
        except Exception:
            return "skipped"

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Return the total number of terms in the store."""
        try:
            return self._get_store()._collection.count()
        except Exception:
            return 0

    def get_all_for_specialty(self, specialty: str) -> list[dict[str, Any]]:
        """Return all stored terms for a given specialty via metadata filter."""
        store = self._get_store()
        try:
            result = store._collection.get(
                where={"specialty": {"$eq": specialty}},
                include=["metadatas", "documents"],
            )
            items = []
            for meta, doc in zip(result.get("metadatas", []), result.get("documents", [])):
                entry = dict(meta)
                entry["_page_content"] = doc
                items.append(entry)
            return items
        except Exception:
            return []
