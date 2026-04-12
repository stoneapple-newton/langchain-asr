# ASR Deep Agents Stage

This stage adds a Deep Agents supervisor plus 20 ASR specialist subagents in
[01_asr_quality_swarm.py](/C:/Users/Newto/Documents/project/test-langchain/deep_research/asr/stage_07_deep_agents/01_asr_quality_swarm.py).

## What It Contains

- one Deep Agents supervisor built with `create_deep_agent(...)`
- 20 ASR-focused subagents registered through the Deep Agents `task` tool
- shared transcript-analysis tools for:
  - transcript overview
  - low-confidence span detection
  - diarization anomaly detection
  - overlap detection
  - glossary extraction
  - filler analysis
  - numeric/entity verification
  - markdown export

## Specialist Agents

1. `audio_intake_agent`
2. `segmentation_agent`
3. `silence_hallucination_agent`
4. `multi_channel_router_agent`
5. `speaker_count_agent`
6. `diarization_repair_agent`
7. `overlap_resolution_agent`
8. `backchannel_merge_agent`
9. `confidence_triage_agent`
10. `phonetic_correction_agent`
11. `acronym_normalizer_agent`
12. `glossary_guard_agent`
13. `punctuation_restoration_agent`
14. `disfluency_policy_agent`
15. `code_switch_audit_agent`
16. `timestamp_alignment_agent`
17. `numeric_entity_verifier_agent`
18. `context_grounding_agent`
19. `severity_review_agent`
20. `redaction_review_agent`

## Install

The official Deep Agents docs describe installation as:

```powershell
uv add deepagents
```

Reference:
- [deepagents package reference](https://reference.langchain.com/python/deepagents/)
- [create_deep_agent reference](https://reference.langchain.com/python/deepagents/graph/create_deep_agent)

## Example

```python
from pathlib import Path
import importlib.util

path = Path("deep_research/asr/stage_07_deep_agents/01_asr_quality_swarm.py")
spec = importlib.util.spec_from_file_location("asr_quality_swarm", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

agent = module.build_asr_quality_swarm()
result = agent.invoke(
    module.example_request(),
    config={"configurable": {"thread_id": "asr-quality-demo"}},
)
```

Use the shared `asr` profile in `config/` to control the underlying model.
