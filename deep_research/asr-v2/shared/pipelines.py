from __future__ import annotations

from typing_extensions import TypedDict

from langgraph.graph import END, START, StateGraph

from shared.transcript_utils import (
    TranscriptDocument,
    analyze_transcript,
    improve_readability,
    load_transcript,
    repair_diarization,
    save_enhanced_outputs,
)


class CleanupState(TypedDict):
    input_path: str
    doc: TranscriptDocument
    analysis_before: dict
    analysis_after: dict


def load_node(state: CleanupState) -> dict:
    doc = load_transcript(state["input_path"])
    return {"doc": doc, "analysis_before": analyze_transcript(doc)}


def repair_node(state: CleanupState) -> dict:
    repaired = repair_diarization(state["doc"])
    return {"doc": repaired, "analysis_after": analyze_transcript(repaired)}


def build_diarization_cleanup_graph():
    builder = StateGraph(CleanupState)
    builder.add_node("load", load_node)
    builder.add_node("repair", repair_node)
    builder.add_edge(START, "load")
    builder.add_edge("load", "repair")
    builder.add_edge("repair", END)
    return builder.compile()


def run_rule_based_cleanup(input_path: str, output_dir: str) -> dict[str, object]:
    original_doc = load_transcript(input_path)
    repaired_doc = improve_readability(repair_diarization(original_doc))
    before = analyze_transcript(original_doc)
    after = analyze_transcript(repaired_doc)
    output_paths = save_enhanced_outputs(repaired_doc, input_path, output_dir)
    return {
        "original_doc": original_doc,
        "repaired_doc": repaired_doc,
        "analysis_before": before,
        "analysis_after": after,
        "output_paths": output_paths,
    }
