"""
Stage 7, File 2: Run Benchmarks
================================
CONCEPT: Systematic evaluation of all summary implementations.

Run each stage implementation against the evaluation dataset
and compare metrics to observe quality progression.

Key concepts:
  - Automated benchmark runner
  - ROUGE scores
  - LLM-as-judge evaluation
  - Comparison reporting

Run this file:
  uv run deep_research/summary_agents/stage_07_evaluation/02_run_benchmarks.py
"""

import json
from pathlib import Path
from typing import Any
from dataclasses import dataclass, field, asdict


# ---------------------------------------------------------------------------
# Benchmark Result Types
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    """Result of benchmarking one implementation on one example."""
    implementation: str  # e.g., "stage_01_basic"
    example_id: str
    
    # Scores
    rouge1_f: float = 0.0
    rouge2_f: float = 0.0
    rougeL_f: float = 0.0
    llm_score: float = 0.0  # 0-10
    
    # Element coverage
    required_elements_found: int = 0
    required_elements_total: int = 0
    
    # Metadata
    latency_ms: int = 0
    token_usage: int = 0
    error: str | None = None
    
    @property
    def element_coverage(self) -> float:
        """Percentage of required elements found."""
        if self.required_elements_total == 0:
            return 0.0
        return self.required_elements_found / self.required_elements_total
    
    @property
    def composite_score(self) -> float:
        """Overall composite score (0-100)."""
        # Weighted combination
        rouge_avg = (self.rouge1_f + self.rouge2_f + self.rougeL_f) / 3
        return (
            rouge_avg * 30 +           # ROUGE: 30%
            (self.llm_score / 10) * 40 +  # LLM: 40%
            self.element_coverage * 30    # Elements: 30%
        )


@dataclass
class BenchmarkReport:
    """Complete benchmark report across all implementations."""
    dataset_name: str
    timestamp: str
    results: list[BenchmarkResult] = field(default_factory=list)
    
    def by_implementation(self) -> dict[str, list[BenchmarkResult]]:
        """Group results by implementation."""
        grouped = {}
        for r in self.results:
            grouped.setdefault(r.implementation, []).append(r)
        return grouped
    
    def summary_stats(self) -> dict[str, Any]:
        """Calculate summary statistics."""
        by_impl = self.by_implementation()
        
        stats = {}
        for impl, results in by_impl.items():
            scores = [r.composite_score for r in results if not r.error]
            stats[impl] = {
                "count": len(results),
                "avg_composite": sum(scores) / len(scores) if scores else 0,
                "avg_rouge1": sum(r.rouge1_f for r in results) / len(results),
                "avg_llm_score": sum(r.llm_score for r in results) / len(results),
                "avg_coverage": sum(r.element_coverage for r in results) / len(results),
            }
        return stats
    
    def save(self, path: str | Path) -> None:
        """Save report to JSON."""
        data = {
            "dataset_name": self.dataset_name,
            "timestamp": self.timestamp,
            "results": [asdict(r) for r in self.results],
            "summary": self.summary_stats(),
        }
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Mock Implementations (for demonstration)
# ---------------------------------------------------------------------------

# These would import actual implementations in production
MOCK_IMPLEMENTATIONS = {
    "stage_01_basic": lambda text: "Basic summary of meeting about JWT and WCAG.",
    "stage_01_structured": lambda text: """# Meeting Summary
## Overview
Technical discussion
## Key Points
- JWT authentication
- WCAG compliance
## Action Items
- Research JWT options""",
    "stage_02_agent": lambda text: """# Meeting Summary

## Overview
Team discussed authentication and accessibility.

## Decisions
- Use JWT for API auth
- WCAG 2.1 AA required

## Action Items
1. Research JWT libraries (Alice)
2. Prepare WCAG checklist (Bob)""",
    "stage_03_langgraph": lambda text: """# Technical Planning Meeting

## Overview
Q4 planning session covering authentication strategy and accessibility compliance.

## Key Discussion Points
1. JWT selected for API authentication
2. WCAG 2.1 AA compliance required
3. Node.js backend evaluation

## Decisions
- JWT for authentication (secure, scalable)
- WCAG 2.1 AA mandatory for all features
- Node.js preferred for I/O operations

## Action Items
- [ ] Alice: Research JWT libraries by Nov 15
- [ ] Bob: Prepare WCAG checklist  
- [ ] Carol: Schedule tech decision meeting""",
    "stage_04_rag": lambda text: """# Nexus Platform - Technical Planning (Nov 12)

## Context
Continuing Q4 planning discussions from Nov 5. Previous meeting established 
authentication as priority and ruled out OAuth.

## Key Discussion Points
1. **JWT Authentication** - Selected for API security
   - Continues discussion from Oct 15 architecture review
   - Refresh token strategy to be defined
   
2. **WCAG Compliance** - Accessibility requirements
   - Related to Oct 22 standards review
   - axe-core integration planned
   
3. **Backend Technology** - Node.js vs Python
   - Builds on Oct 29 evaluation
   - Decision needed for Q4 deliverables

## Decisions
- JWT implementation (reference: RFC 7519)
- WCAG 2.1 AA compliance mandatory
- Node.js for I/O-bound services

## Action Items
- [ ] Alice: JWT library research (due: Nov 15)
- [ ] Bob: WCAG checklist (due: Nov 15)  
- [ ] Carol: Tech decision meeting (due: Nov 12)""",
}


# ---------------------------------------------------------------------------
# Evaluation Functions
# ---------------------------------------------------------------------------

def calculate_rouge(reference: str, generated: str) -> dict[str, float]:
    """Calculate ROUGE scores."""
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        scores = scorer.score(reference, generated)
        return {
            "rouge1_f": scores["rouge1"].fmeasure,
            "rouge2_f": scores["rouge2"].fmeasure,
            "rougeL_f": scores["rougeL"].fmeasure,
        }
    except ImportError:
        # Fallback: simple word overlap
        ref_words = set(reference.lower().split())
        gen_words = set(generated.lower().split())
        overlap = len(ref_words & gen_words)
        score = overlap / max(len(ref_words), 1)
        return {"rouge1_f": score, "rouge2_f": score, "rougeL_f": score}


def check_elements(summary: str, required: list[str]) -> tuple[int, int]:
    """Check which required elements are present in summary."""
    found = sum(1 for elem in required if elem.lower() in summary.lower())
    return found, len(required)


def llm_judge_score(reference: str, generated: str) -> float:
    """Use LLM to score summary quality (0-10)."""
    # In production, would call LLM
    # For demo, return simulated scores based on length similarity
    ref_len = len(reference.split())
    gen_len = len(generated.split())
    
    # Simple heuristic: closer length = better (very approximate)
    len_ratio = min(ref_len, gen_len) / max(ref_len, gen_len)
    base_score = 5 + (len_ratio * 3)  # 5-8 range
    
    # Boost for structured content markers
    if "#" in generated:
        base_score += 0.5
    if "##" in generated:
        base_score += 0.5
    if "- [ ]" in generated or "Action" in generated:
        base_score += 0.5
    
    return min(10, base_score)


# ---------------------------------------------------------------------------
# Benchmark Runner
# ---------------------------------------------------------------------------

def run_benchmark(dataset_path: str | Path) -> BenchmarkReport:
    """Run benchmarks on all implementations."""
    from datetime import datetime
    
    # Load dataset
    dataset_data = json.loads(Path(dataset_path).read_text())
    examples = dataset_data.get("examples", [])
    
    report = BenchmarkReport(
        dataset_name=dataset_data.get("name", "unknown"),
        timestamp=datetime.now().isoformat(),
    )
    
    print("=" * 70)
    print("  STAGE 7.2: RUNNING BENCHMARKS")
    print("=" * 70)
    print(f"\nDataset: {report.dataset_name}")
    print(f"Examples: {len(examples)}")
    print(f"Implementations: {len(MOCK_IMPLEMENTATIONS)}\n")
    
    for impl_name, impl_func in MOCK_IMPLEMENTATIONS.items():
        print(f"\nBenchmarking: {impl_name}")
        print("-" * 50)
        
        for ex in examples:
            ex_id = ex.get("id", "unknown")
            print(f"  Example: {ex_id}...", end=" ")
            
            try:
                # Generate summary
                transcript = ex.get("transcript_text", "")
                if not transcript:
                    # Load from path
                    transcript_path = ex.get("transcript_path", "")
                    if transcript_path:
                        # In production, would load actual file
                        transcript = "Sample transcript content for " + ex_id
                
                generated = impl_func(transcript)
                reference = ex.get("reference_summary", "")
                required = ex.get("required_elements", [])
                
                # Calculate scores
                rouge = calculate_rouge(reference, generated)
                elements_found, elements_total = check_elements(generated, required)
                llm_score = llm_judge_score(reference, generated)
                
                result = BenchmarkResult(
                    implementation=impl_name,
                    example_id=ex_id,
                    rouge1_f=rouge["rouge1_f"],
                    rouge2_f=rouge["rouge2_f"],
                    rougeL_f=rouge["rougeL_f"],
                    llm_score=llm_score,
                    required_elements_found=elements_found,
                    required_elements_total=elements_total,
                )
                
                report.results.append(result)
                print(f"✓ Score: {result.composite_score:.1f}")
                
            except Exception as e:
                result = BenchmarkResult(
                    implementation=impl_name,
                    example_id=ex_id,
                    error=str(e),
                )
                report.results.append(result)
                print(f"✗ Error: {e}")
    
    return report


# ---------------------------------------------------------------------------
# Report Generation
# ---------------------------------------------------------------------------

def print_comparison_table(stats: dict[str, Any]) -> None:
    """Print comparison table of implementation stats."""
    print("\n" + "=" * 70)
    print("  BENCHMARK RESULTS")
    print("=" * 70)
    
    # Header
    print(f"\n{'Implementation':<20} {'Composite':>10} {'ROUGE-1':>10} {'LLM':>10} {'Coverage':>10}")
    print("-" * 70)
    
    # Sort by composite score
    sorted_impls = sorted(
        stats.items(),
        key=lambda x: x[1].get("avg_composite", 0),
        reverse=True
    )
    
    for impl, stat in sorted_impls:
        print(
            f"{impl:<20} "
            f"{stat.get('avg_composite', 0):>10.1f} "
            f"{stat.get('avg_rouge1', 0):>10.2f} "
            f"{stat.get('avg_llm_score', 0):>10.1f} "
            f"{stat.get('avg_coverage', 0)*100:>9.0f}%"
        )
    
    print("\n" + "-" * 70)
    print("Score components:")
    print("  Composite: Weighted combination (ROUGE 30%, LLM 40%, Coverage 30%)")
    print("  ROUGE-1:   Word overlap with reference")
    print("  LLM:       Semantic quality score (0-10)")
    print("  Coverage:  % of required elements found")


def main():
    # Check for dataset
    dataset_path = Path(__file__).parent / "summary_evaluation_dataset.json"
    if not dataset_path.exists():
        print(f"Dataset not found. Run 01_create_dataset.py first.")
        return
    
    # Run benchmarks
    report = run_benchmark(dataset_path)
    
    # Print results
    stats = report.summary_stats()
    print_comparison_table(stats)
    
    # Save report
    report_path = Path(__file__).parent / "benchmark_report.json"
    report.save(report_path)
    
    print(f"\n\nReport saved to: {report_path}")
    
    # Key findings
    print("\n" + "=" * 70)
    print("  KEY FINDINGS")
    print("=" * 70)
    
    best_impl = max(stats.items(), key=lambda x: x[1].get("avg_composite", 0))
    print(f"\n🏆 Best performing: {best_impl[0]}")
    print(f"   Composite score: {best_impl[1]['avg_composite']:.1f}")
    
    # Quality progression
    print("\n📈 Quality Progression:")
    stages = ["stage_01_basic", "stage_01_structured", "stage_02_agent", 
              "stage_03_langgraph", "stage_04_rag"]
    for stage in stages:
        if stage in stats:
            score = stats[stage]["avg_composite"]
            bar = "█" * int(score / 5)
            print(f"   {stage:<25} {score:>5.1f} {bar}")


if __name__ == "__main__":
    main()
