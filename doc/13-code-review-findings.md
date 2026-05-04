# Code Review Findings

## Review Summary

The codebase is coherent as an educational workspace. The strongest reusable
areas are `config/`, ASR-v2 shared utilities, PII utilities, translation
utilities, and medical verification helpers. The largest maintainability risks
come from script-style imports, model clients created at module import time,
duplicate transcript contracts, and uneven test inclusion across feature
tracks.

## Finding 1: Some LLM Clients Are Created At Import Time

Severity: Medium

Examples:

- `deep_research/asr_comparison/02_langgraph_agent.py` creates `_llm` and
  `_chain` at module import time.
- `deep_research/transcription_correction_agent/agent.py` creates `_llm` and a
  singleton `correction_graph` at module import time.
- Several tutorial scripts execute model setup or example logic at import time.

Impact:

Imports can fail when Ollama is not running, credentials are missing, or an
optional provider package is not installed. This also makes tests rely on
monkeypatching and fake modules.

Recommendation:

Move model construction into `build_*()` functions, `main()`, or lazy helpers.
Allow graph builders to accept an LLM or profile name as an argument.

## Finding 2: Path Injection And Non-Package Imports Are Common

Severity: Medium

Examples:

- ASR comparison adds `deep_research/asr-v2` to `sys.path` and imports
  `shared.*`.
- Transcription correction imports `from state import ...` and `from tools.*`
  after modifying `sys.path`.
- Translation dynamically imports ASR-v2 helpers because `asr-v2` contains a
  hyphen.

Impact:

These patterns work for runnable tutorials but are fragile in larger
applications. They can collide with unrelated `shared`, `state`, or `tools`
modules and make reuse harder.

Recommendation:

For reusable modules, move shared code under package-safe directories, for
example `deep_research/asr_v2/` or a common `deep_research/shared/` package.
Use package-relative imports in feature packages.

## Finding 3: Transcript Data Models Are Duplicated Across Tracks

Severity: Medium

Examples:

- ASR-v2 defines `WordToken`, `TranscriptSegment`, and `TranscriptDocument`.
- Summary agents define a separate `TranscriptSegment` and `MeetingTranscript`.
- Medical verification defines its own transcript utilities and correction
  records.
- Original ASR examples often use raw dictionaries.

Impact:

Duplicated contracts increase conversion work and make new feature development
less predictable.

Recommendation:

Use ASR-v2 dataclasses as the default transcript contract for new work. Add
adapters only when a feature needs domain-specific fields.

## Finding 4: Test Discovery Does Not Include All Track-Local Tests By Default

Severity: Low to Medium

`pyproject.toml` sets pytest paths to `tests` and `deep_research/asr/tests`.
Other track-local tests, such as ASR-v2, ASR comparison, and diarization tests,
must be run explicitly.

Impact:

Developers can miss regressions in active feature tracks if they only run the
default pytest command.

Recommendation:

Either add active track-local test folders to `testpaths` or document a standard
full test command that includes them.

## Finding 5: High-Stakes Medical Correction Applies In-Place In One Workflow

Severity: Medium

`deep_research/asr_medical_verification/workflow.py` saves the corrected
document back to the transcript input path and writes a sidecar.

Impact:

This is convenient for workflow tests but risky for production-like usage,
especially in a medical domain.

Recommendation:

Add an explicit output path or `overwrite` option. Default production-oriented
commands should write to a new output file and keep the original transcript
unchanged.

## Finding 6: Optional Dependency Behavior Is Inconsistent

Severity: Low

Some optional dependencies are handled gracefully, such as ROUGE and Deep
Agents tests. Other optional runtime tools, such as audio extraction or
provider availability, are handled through broader fallbacks.

Impact:

Developers may not know whether a skipped capability is expected, unavailable,
or failing.

Recommendation:

Document optional dependencies per feature and expose clear capability checks
for audio, Deep Agents, and evaluation extras.

## Strengths

- Shared provider config is well tested and keeps tutorial scripts
  provider-agnostic.
- ASR-v2 utilities provide clear reusable transcript contracts.
- PII redaction has strong deterministic span handling and evaluation helpers.
- Translation utilities enforce segment, speaker, and timestamp preservation.
- Benchmark runners increasingly report failures without aborting all variants.
- The staged structure is useful for learning and for comparing framework
  layers.

## Suggested Refactor Backlog

1. Add a full-suite test command or expand pytest `testpaths`.
2. Lazily construct LLM clients in reusable graph modules.
3. Create a package-safe ASR-v2 namespace or shared transcript package.
4. Add explicit output path controls to medical correction workflows.
5. Consolidate transcript adapter utilities across ASR, summary, translation,
   and medical verification tracks.
6. Add docs for optional dependencies and capability fallbacks.
