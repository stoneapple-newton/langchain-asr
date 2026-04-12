# ASR Quality Agent Tutorial

This staged tutorial mirrors the style of `deep_research_agent`, but focuses on
post-processing WhisperX-like ASR JSON to improve diarization quality and human
readability.

## Stages

1. `stage_01_basics/01_inspect_whisperx_json.py`
2. `stage_01_basics/02_build_canonical_transcript.py`
3. `stage_02_tools/01_rule_based_cleanup.py`
4. `stage_02_tools/02_llm_readability_editor.py`
5. `stage_03_langgraph/01_diarization_cleanup_graph.py`
6. `stage_03_langgraph/02_readability_revision_graph.py`
7. `stage_04_agent/01_asr_quality_agent.py`
8. `stage_05_production/01_batch_asr_quality_runner.py`

## Run commands

From `C:\Users\Newto\Documents\project\test-langchain`:

```powershell
uv run deep_research/asr-v2/stage_01_basics/01_inspect_whisperx_json.py
uv run deep_research/asr-v2/stage_01_basics/02_build_canonical_transcript.py
uv run deep_research/asr-v2/stage_02_tools/01_rule_based_cleanup.py
uv run deep_research/asr-v2/stage_02_tools/02_llm_readability_editor.py
uv run deep_research/asr-v2/stage_03_langgraph/01_diarization_cleanup_graph.py
uv run deep_research/asr-v2/stage_03_langgraph/02_readability_revision_graph.py
uv run deep_research/asr-v2/stage_04_agent/01_asr_quality_agent.py
uv run deep_research/asr-v2/stage_05_production/01_batch_asr_quality_runner.py
```

## Pytest

Run the deterministic ASR suite from `C:\Users\Newto\Documents\project\test-langchain`:

```powershell
uv run --with pytest pytest deep_research/asr-v2/tests -q
```

The pytest suite covers the shared transcript utilities plus the deterministic
cleanup graph. The LLM-backed stages are still intended as manual integration
runs because they depend on a reachable chat provider configured for the
`asr_v2` profile in the shared [config package](/C:/Users/Newto/Documents/project/test-langchain/config/README.md).

## Input shape

The loader expects WhisperX-like JSON with:

- top-level `segments`
- segment `start`, `end`, `text`
- optional segment `speaker`
- optional `words` list with per-word timing and speaker labels

The parser is intentionally generic-first. If you later provide a real export
sample, the adapter can be tightened around that exact schema.

## Outputs

Later stages write files under `outputs/<input-stem>/`:

- `<input>.enhanced.json`
- `<input>.transcript.md`

The JSON remains WhisperX-like and preserves timing plus speaker annotations.
