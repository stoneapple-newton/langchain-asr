"""
Evaluation utilities for summary quality assessment.

Provides tools for evaluating summary quality using:
- ROUGE scores (overlap with reference)
- LLM-as-judge for semantic similarity
- Custom metrics for meeting summaries (action item recall, etc.)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# Import config for LLM creation
def _get_llm():
    """Lazy import to avoid circular dependencies."""
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from config import create_chat_model
    return create_chat_model(temperature=0)


class SummaryQualityScore(BaseModel):
    """Structured output for LLM-based quality evaluation."""

    completeness: int = Field(
        description="Coverage of key points (1-10)", ge=1, le=10
    )
    accuracy: int = Field(
        description="Factual correctness (1-10)", ge=1, le=10
    )
    conciseness: int = Field(
        description="Brevity without losing information (1-10)", ge=1, le=10
    )
    action_items: int = Field(
        description="Action items correctly identified (1-10)", ge=1, le=10
    )
    overall: int = Field(
        description="Overall quality score (1-10)", ge=1, le=10
    )
    explanation: str = Field(
        description="Brief explanation of the scoring"
    )


@dataclass
class EvaluationResult:
    """Result of evaluating a summary against reference."""

    rouge1_f: float = 0.0
    rouge2_f: float = 0.0
    rougeL_f: float = 0.0
    llm_completeness: int = 0
    llm_accuracy: int = 0
    llm_conciseness: int = 0
    llm_action_items: int = 0
    llm_overall: int = 0
    llm_explanation: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def rouge_average(self) -> float:
        """Average of ROUGE F1 scores."""
        return (self.rouge1_f + self.rouge2_f + self.rougeL_f) / 3

    @property
    def llm_average(self) -> float:
        """Average of LLM quality scores."""
        return (
            self.llm_completeness
            + self.llm_accuracy
            + self.llm_conciseness
            + self.llm_action_items
        ) / 4

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "rouge1_f": round(self.rouge1_f, 4),
            "rouge2_f": round(self.rouge2_f, 4),
            "rougeL_f": round(self.rougeL_f, 4),
            "rouge_average": round(self.rouge_average, 4),
            "llm_completeness": self.llm_completeness,
            "llm_accuracy": self.llm_accuracy,
            "llm_conciseness": self.llm_conciseness,
            "llm_action_items": self.llm_action_items,
            "llm_overall": self.llm_overall,
            "llm_average": round(self.llm_average, 4),
            "llm_explanation": self.llm_explanation,
            "metadata": self.metadata,
        }


class SummaryEvaluator:
    """Evaluates meeting summary quality."""

    def __init__(self, use_llm_judge: bool = True):
        self.use_llm_judge = use_llm_judge
        self._llm = None
        self._eval_chain = None

    def _get_eval_chain(self):
        """Lazy initialization of evaluation chain."""
        if self._eval_chain is None:
            llm = _get_llm()
            parser = JsonOutputParser(pydantic_object=SummaryQualityScore)
            prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    "You are an expert evaluator of meeting summaries. "
                    "Score the generated summary against the reference summary.\n\n"
                    "{format_instructions}"
                ),
                (
                    "human",
                    "Reference Summary (ground truth):\n{reference}\n\n"
                    "Generated Summary (to evaluate):\n{generated}\n\n"
                    "Provide your evaluation as JSON."
                ),
            ]).partial(format_instructions=parser.get_format_instructions())
            self._eval_chain = prompt | llm | parser
        return self._eval_chain

    def evaluate(
        self,
        generated_summary: str,
        reference_summary: str,
        transcript: str | None = None,
    ) -> EvaluationResult:
        """
        Evaluate a generated summary against a reference.

        Args:
            generated_summary: The summary to evaluate
            reference_summary: The ground truth reference
            transcript: Optional original transcript for additional context

        Returns:
            EvaluationResult with scores
        """
        result = EvaluationResult()

        # Calculate ROUGE scores (optional - requires rouge-score package)
        try:
            from rouge_score import rouge_scorer

            scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
            scores = scorer.score(reference_summary, generated_summary)
            result.rouge1_f = scores["rouge1"].fmeasure
            result.rouge2_f = scores["rouge2"].fmeasure
            result.rougeL_f = scores["rougeL"].fmeasure
        except ImportError:
            # ROUGE not available, skip
            pass

        # LLM-as-judge evaluation
        if self.use_llm_judge:
            try:
                chain = self._get_eval_chain()
                eval_result = chain.invoke({
                    "reference": reference_summary,
                    "generated": generated_summary,
                })
                result.llm_completeness = eval_result.get("completeness", 0)
                result.llm_accuracy = eval_result.get("accuracy", 0)
                result.llm_conciseness = eval_result.get("conciseness", 0)
                result.llm_action_items = eval_result.get("action_items", 0)
                result.llm_overall = eval_result.get("overall", 0)
                result.llm_explanation = eval_result.get("explanation", "")
            except Exception as e:
                result.metadata["llm_eval_error"] = str(e)

        return result


@dataclass
class EvaluationExample:
    """A single example for evaluation dataset."""

    name: str
    transcript_path: str
    reference_summary: str
    metadata: dict[str, Any] = field(default_factory=dict)


class EvaluationDataset:
    """Dataset of examples for benchmarking summary agents."""

    def __init__(self, name: str, examples: list[EvaluationExample] | None = None):
        self.name = name
        self.examples = examples or []

    def add_example(self, example: EvaluationExample) -> None:
        """Add an example to the dataset."""
        self.examples.append(example)

    def save(self, path: str | Path) -> None:
        """Save dataset to JSON file."""
        data = {
            "name": self.name,
            "examples": [
                {
                    "name": ex.name,
                    "transcript_path": ex.transcript_path,
                    "reference_summary": ex.reference_summary,
                    "metadata": ex.metadata,
                }
                for ex in self.examples
            ],
        }
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "EvaluationDataset":
        """Load dataset from JSON file."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        examples = [
            EvaluationExample(
                name=ex["name"],
                transcript_path=ex["transcript_path"],
                reference_summary=ex["reference_summary"],
                metadata=ex.get("metadata", {}),
            )
            for ex in data["examples"]
        ]
        return cls(name=data["name"], examples=examples)


def create_evaluation_dataset(output_path: str | Path | None = None) -> EvaluationDataset:
    """
    Create the default evaluation dataset for summary agents.

    This includes a few hand-crafted examples with reference summaries
    for benchmarking different summarization approaches.
    """
    dataset = EvaluationDataset(name="summary_agents_benchmark_v1")

    # Example 1: Simple meeting (using sample transcript)
    dataset.add_example(
        EvaluationExample(
            name="q4_planning_meeting",
            transcript_path="deep_research/summary_agents/shared/sample_data/meeting_sample.json",
            reference_summary="""# Q4 Planning Meeting Summary

## Participants
- SPEAKER_00 (likely PM/Lead)
- SPEAKER_01 (engineer)
- SPEAKER_02 (another team member)

## Key Points
1. Reviewed Q4 roadmap priorities and shipping timeline
2. Discussed API integration challenges and JWT authentication requirements
3. Identified WCAG compliance as a key deliverable
4. Explored Node.js vs Python backend options for the service

## Decisions Made
- Focus on authentication/authorization (JWT) as priority for next sprint
- Team will evaluate WCAG 2.1 AA compliance requirements
- Need to finalize tech stack decision (Node.js) by end of week

## Action Items
- [ ] SPEAKER_01: Research JWT libraries and present options
- [ ] Team: Review WCAG checklist and identify gaps
- [ ] SPEAKER_00: Schedule follow-up for tech stack decision""",
            metadata={
                "topic": "Q4 Planning",
                "duration_minutes": 3,
                "num_speakers": 3,
                "difficulty": "easy",
            },
        )
    )

    # Example 2: Technical discussion (simulated)
    dataset.add_example(
        EvaluationExample(
            name="architecture_review",
            transcript_path="deep_research/summary_agents/shared/sample_data/meeting_sample.json",
            reference_summary="""# Architecture Review: Authentication Service

## Overview
Team discussed implementation approach for the new authentication microservice.

## Key Discussion Points
- JWT token strategy: access vs refresh tokens
- Node.js ecosystem advantages for async I/O
- WCAG accessibility requirements for login flows
- SLA targets: 99.9% uptime, <100ms response time

## Decisions
1. Use Node.js with Express for the service
2. Implement JWT with 15-min access / 7-day refresh token expiry
3. WCAG 2.1 AA compliance required for all auth flows

## Open Questions
- Database choice (PostgreSQL vs Redis for sessions)
- Rate limiting strategy""",
            metadata={
                "topic": "Architecture Review",
                "duration_minutes": 45,
                "num_speakers": 4,
                "difficulty": "medium",
            },
        )
    )

    if output_path:
        dataset.save(output_path)

    return dataset


if __name__ == "__main__":
    # Demo: Create and save the default dataset
    dataset_path = Path(__file__).parent / "sample_data" / "evaluation_dataset.json"
    dataset = create_evaluation_dataset(dataset_path)
    print(f"Created evaluation dataset: {dataset_path}")
    print(f"Examples: {len(dataset.examples)}")
    for ex in dataset.examples:
        print(f"  - {ex.name}: {ex.metadata.get('topic', 'Unknown')}")
