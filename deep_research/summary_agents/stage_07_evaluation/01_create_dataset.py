"""
Stage 7, File 1: Create Evaluation Dataset
===========================================
CONCEPT: Building a dataset for benchmarking summary quality.

To measure improvement across implementations, we need:
- Example transcripts
- Reference (ground truth) summaries
- Evaluation criteria

Key concepts:
  - Dataset structure
  - Reference summary creation
  - Quality criteria definition

Run this file:
  uv run deep_research/summary_agents/stage_07_evaluation/01_create_dataset.py
"""

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Dataset Schema
# ---------------------------------------------------------------------------

class EvaluationExample(BaseModel):
    """A single example for evaluation."""
    
    id: str = Field(description="Unique identifier")
    name: str = Field(description="Human-readable name")
    
    # Input
    transcript_path: str = Field(description="Path to transcript file")
    transcript_text: str | None = Field(default=None, description="Full transcript text")
    
    # Expected output
    reference_summary: str = Field(description="Ground truth summary")
    
    # Criteria
    required_elements: list[str] = Field(
        default_factory=list,
        description="Elements that must appear in summary"
    )
    
    # Metadata
    difficulty: str = Field(
        default="medium",
        description="easy, medium, or hard"
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationDataset(BaseModel):
    """Collection of evaluation examples."""
    
    name: str = Field(description="Dataset name")
    version: str = Field(default="1.0.0")
    description: str = Field(default="")
    examples: list[EvaluationExample] = Field(default_factory=list)
    
    def save(self, path: str | Path) -> None:
        """Save dataset to JSON file."""
        Path(path).write_text(
            self.model_dump_json(indent=2),
            encoding="utf-8"
        )
    
    @classmethod
    def load(cls, path: str | Path) -> "EvaluationDataset":
        """Load dataset from JSON file."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)


# ---------------------------------------------------------------------------
# Create Dataset
# ---------------------------------------------------------------------------

def create_summary_evaluation_dataset() -> EvaluationDataset:
    """Create the default evaluation dataset for summary agents."""
    
    dataset = EvaluationDataset(
        name="summary_agents_evaluation",
        version="1.0.0",
        description="Evaluation dataset for meeting summary quality",
    )
    
    # Example 1: Technical planning meeting
    dataset.examples.append(
        EvaluationExample(
            id="eval_001",
            name="api_architecture_review",
            transcript_path="deep_research/summary_agents/shared/sample_data/meeting_sample.json",
            reference_summary="""# API Architecture Review - Meeting Summary

## Overview
Technical planning session focused on Q4 deliverables for the Nexus Platform. 
Team discussed authentication mechanisms (JWT), accessibility compliance (WCAG), 
and backend technology choices.

## Key Discussion Points

### Authentication Strategy
- JWT (JSON Web Tokens) selected for API authentication
- Discussion of token refresh mechanisms
- Security considerations for token storage

### Accessibility Compliance  
- WCAG 2.1 AA compliance required for all new features
- Frontend team to integrate axe-core testing
- Design system needs color contrast updates

### Backend Technology
- Evaluation of Node.js vs Python for microservices
- Node.js preferred for I/O-bound operations
- Team training needs identified for async patterns

## Decisions Made
1. **Authentication**: Proceed with JWT implementation
2. **Accessibility**: WCAG 2.1 AA compliance mandatory
3. **Backend**: Node.js selected for new services

## Action Items
1. Research JWT libraries and present options (Alice)
2. Prepare WCAG compliance checklist (Bob)
3. Schedule backend tech decision meeting (Carol)

## Next Steps
- Finalize authentication library selection
- Begin accessibility audit preparation
- Draft Node.js service migration plan
""",
            required_elements=[
                "JWT",
                "authentication",
                "WCAG",
                "accessibility",
                "Node.js",
                "action items",
                "decisions"
            ],
            difficulty="medium",
            metadata={
                "topic": "API Architecture",
                "duration_minutes": 3,
                "speakers": 3,
                "meeting_type": "technical_planning",
            },
        )
    )
    
    # Example 2: Project standup (simulated)
    dataset.examples.append(
        EvaluationExample(
            id="eval_002",
            name="daily_standup",
            transcript_path="deep_research/summary_agents/shared/sample_data/meeting_sample.json",
            reference_summary="""# Daily Standup Summary

## Team Updates

### Alice (Senior Engineer)
- **Yesterday**: Completed API integration work
- **Today**: Working on test coverage
- **Blockers**: None

### Bob (Product Manager)
- **Yesterday**: Reviewed roadmap with stakeholders
- **Today**: Finalizing Q4 priorities
- **Blockers**: Waiting on budget approval

### Carol (Tech Lead)
- **Yesterday**: Code reviews for team PRs
- **Today**: Architecture planning session
- **Blockers**: Need decision on database choice

## Blockers Requiring Attention
1. Budget approval blocking Q4 planning (Bob)
2. Database decision needed for Carol's work

## Team Velocity
- On track for sprint goals
- 2 PRs pending review
- No critical bugs reported
""",
            required_elements=[
                "standup",
                "blockers",
                "yesterday",
                "today",
                "team velocity"
            ],
            difficulty="easy",
            metadata={
                "topic": "Daily Standup",
                "duration_minutes": 15,
                "speakers": 3,
                "meeting_type": "standup",
            },
        )
    )
    
    # Example 3: Retrospective (simulated)
    dataset.examples.append(
        EvaluationExample(
            id="eval_003",
            name="sprint_retrospective",
            transcript_path="deep_research/summary_agents/shared/sample_data/meeting_sample.json",
            reference_summary="""# Sprint Retrospective Summary

## What Went Well ✅
- Smooth deployment process with new CI/CD pipeline
- Team collaboration on complex authentication features
- Early detection of WCAG issues allowed time to fix

## What Could Be Improved ⚠️
- Communication between frontend and backend teams
- Late requirement changes caused scope creep
- Testing coverage gaps in API layer

## Action Items for Next Sprint
1. Establish daily sync between frontend and backend leads
2. Implement stricter change control process
3. Add API contract testing to CI pipeline

## Team Sentiment
Positive overall. Team feels progress is being made on technical debt while 
delivering features. Main concern is maintaining quality under timeline pressure.

## Metrics
- Velocity: 23 points (planned: 25)
- Bugs introduced: 3 (down from 8 last sprint)
- Cycle time: Improved by 15%
""",
            required_elements=[
                "went well",
                "improved",
                "action items",
                "sentiment",
                "metrics"
            ],
            difficulty="medium",
            metadata={
                "topic": "Sprint Retrospective",
                "duration_minutes": 60,
                "speakers": 5,
                "meeting_type": "retrospective",
            },
        )
    )
    
    return dataset


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  STAGE 7.1: CREATE EVALUATION DATASET")
    print("=" * 70)
    
    dataset = create_summary_evaluation_dataset()
    
    print(f"\nDataset: {dataset.name}")
    print(f"Version: {dataset.version}")
    print(f"Description: {dataset.description}")
    print(f"\nExamples: {len(dataset.examples)}")
    
    for ex in dataset.examples:
        print(f"\n  {ex.id}: {ex.name}")
        print(f"    Difficulty: {ex.difficulty}")
        print(f"    Required elements: {len(ex.required_elements)}")
        print(f"    Reference length: {len(ex.reference_summary)} chars")
    
    # Save to file
    output_path = Path(__file__).parent / "summary_evaluation_dataset.json"
    dataset.save(output_path)
    
    print(f"\n" + "=" * 70)
    print(f"  Dataset saved to: {output_path}")
    print("=" * 70)
    
    # Verify by loading
    loaded = EvaluationDataset.load(output_path)
    print(f"\nVerification: Loaded {len(loaded.examples)} examples successfully")
    
    print("""
DATASET USAGE:
--------------
```python
from deep_research.summary_agents.stage_07_evaluation import EvaluationDataset

# Load dataset
dataset = EvaluationDataset.load("summary_evaluation_dataset.json")

# Iterate examples
for example in dataset.examples:
    # Run your summarizer
    generated = your_summarizer(example.transcript_path)
    
    # Compare to reference
    scores = evaluate(generated, example.reference_summary)
    
    # Check required elements
    missing = [e for e in example.required_elements if e not in generated]
```
""")


if __name__ == "__main__":
    main()
