# ASR Deep Agents Stage

This stage now contains two Deep Agents workflows:

- **ASR quality swarm** with 20 specialist subagents in
  [`01_asr_quality_swarm.py`](./01_asr_quality_swarm.py)
- **Medical extraction swarm** for medical terms + medicine names in
  [`02_medical_term_extraction_swarm.py`](./02_medical_term_extraction_swarm.py)

## What It Contains

- Deep Agents supervisors built with `create_deep_agent(...)`
- specialist subagents registered through the Deep Agents `task` tool
- shared transcript-analysis tools for structured extraction and review

## ASR Quality Specialist Agents

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

## Medical Extraction Specialist Agents

1. `clinical_intake_agent`
2. `medical_terms_agent`
3. `medication_names_agent`
4. `ambiguity_resolution_agent`
5. `clinical_summary_agent`

## Install

```bash
uv add deepagents
```

References:
- [deepagents package reference](https://reference.langchain.com/python/deepagents/)
- [create_deep_agent reference](https://reference.langchain.com/python/deepagents/graph/create_deep_agent)

## Example (Medical Extraction)

```python
from pathlib import Path
import importlib.util

path = Path("deep_research/asr/stage_07_deep_agents/02_medical_term_extraction_swarm.py")
spec = importlib.util.spec_from_file_location("medical_swarm", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

agent = module.build_medical_extraction_swarm()
result = agent.invoke(
    module.example_request(),
    config={"configurable": {"thread_id": "asr-medical-demo"}},
)
```

Use the shared `asr` profile in `config/` to control the underlying model.
