# Tests and Quality

## Feature Summary

The test suite verifies deterministic utilities, configuration behavior,
variant contracts, benchmark behavior, and smoke imports for selected stage
scripts.

## Business Outcome

Tests protect the reusable parts of the educational codebase while keeping
model-dependent examples lightweight.

## Scope

Root tests:

- `tests/test_config.py`
- `tests/test_pii_redaction_utils.py`
- `tests/test_asr_deep_agents.py`
- `tests/test_asr_medical_extraction_deep_agents.py`
- `tests/test_asr_medical_verification.py`
- `tests/test_asr_translation_*`

Track-local tests:

- `deep_research/asr/tests/*`
- `deep_research/asr-v2/tests/*`
- `deep_research/asr_comparison/tests/*`
- `deep_research/diarization_improvements/tests/*`

## Current Architecture

Tests generally avoid real model calls by:

- Testing deterministic helper functions directly.
- Monkeypatching or faking config and LLM modules.
- Smoke-importing selected scripts with fake dependencies.
- Verifying optional Deep Agents behavior when the package is unavailable.
- Using local sample data and temporary output folders.

`pyproject.toml` configures pytest to run `tests` and `deep_research/asr/tests`
by default. Other track-local tests are runnable explicitly.

## Key Quality Signals

- Config tests cover default settings, nested env parsing, legacy aliases,
  provider constructor kwargs, OpenRouter headers, missing credentials, and
  unsupported embeddings.
- PII tests cover dataset loading, entity normalization, redaction, regex
  candidates, scoring, and stage imports.
- ASR-v2 tests cover transcript utilities, deterministic pipeline behavior,
  and transcript diff behavior.
- Translation tests cover dataset contracts, structural preservation, scoring,
  output writing, variant contracts, Deep Agents availability, and benchmark
  failure behavior.
- Medical verification tests cover glossary linking, candidate detection,
  correction application, and workflow output writing.

## Dependencies

- pytest
- pytest-mock
- local sample data
- test-time monkeypatching

## Nonfunctional Requirements

- Unit tests should not require network access or real model credentials.
- Deterministic helpers should have direct tests.
- LLM-heavy examples should expose enough pure logic to test without invoking
  models.
- Benchmark tests should verify failure reporting.

## Acceptance Criteria

- `uv run --with pytest pytest -q` runs the configured test suite.
- Feature-specific tests can be run explicitly for deeper coverage.
- New shared helpers include tests.
- Optional dependency behavior is tested when relevant.

## Risks and Open Questions

- The default pytest paths do not include every track-local test folder.
- Some scripts still instantiate LLM clients at import time, making them harder
  to test without monkeypatching.
- Test coverage is uneven across educational tracks.
- Some tests depend on path manipulation because `asr-v2` is not a package-safe
  Python module name.

## Development Guidance

For new code, first isolate deterministic logic behind functions. Then add
tests for those functions before wiring them into LLM scripts. If a stage uses
an optional package, test both the installed and unavailable paths when
practical.
