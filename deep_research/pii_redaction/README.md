# PII Redaction Progression

This track builds the same PII-redaction task across four implementation
styles, then benchmarks them on a shared labeled dataset.

## Stages

1. [stage_01_basics/01_one_shot_prompt_redactor.py](/C:/Users/Newto/Documents/project/test-langchain/deep_research/pii_redaction/stage_01_basics/01_one_shot_prompt_redactor.py)
2. [stage_02_agents/01_langchain_pii_agent.py](/C:/Users/Newto/Documents/project/test-langchain/deep_research/pii_redaction/stage_02_agents/01_langchain_pii_agent.py)
3. [stage_03_langgraph/01_pii_redaction_workflow.py](/C:/Users/Newto/Documents/project/test-langchain/deep_research/pii_redaction/stage_03_langgraph/01_pii_redaction_workflow.py)
4. [stage_04_deep_agents/01_pii_redaction_deep_agent.py](/C:/Users/Newto/Documents/project/test-langchain/deep_research/pii_redaction/stage_04_deep_agents/01_pii_redaction_deep_agent.py)
5. [stage_05_evaluation/01_benchmark_redaction_variants.py](/C:/Users/Newto/Documents/project/test-langchain/deep_research/pii_redaction/stage_05_evaluation/01_benchmark_redaction_variants.py)

## Dataset

The benchmark dataset lives at
[dataset/pii_redaction_dataset.json](/C:/Users/Newto/Documents/project/test-langchain/deep_research/pii_redaction/dataset/pii_redaction_dataset.json)
and includes:

- names
- emails
- phones
- addresses
- SSNs
- DOBs
- credit cards
- IP addresses
- employee IDs
- MRNs
- account and routing numbers
- passport numbers
- one no-PII control sample

## Benchmark Metrics

Each variant is scored on:

- entity precision
- entity recall
- entity F1
- exact redaction rate

The shared utilities normalize exact substring matches and apply canonical
placeholders like `[PERSON]`, `[EMAIL]`, and `[SSN]`.

## Run Commands

```powershell
uv run deep_research/pii_redaction/stage_01_basics/01_one_shot_prompt_redactor.py
uv run deep_research/pii_redaction/stage_02_agents/01_langchain_pii_agent.py
uv run deep_research/pii_redaction/stage_03_langgraph/01_pii_redaction_workflow.py
uv run deep_research/pii_redaction/stage_05_evaluation/01_benchmark_redaction_variants.py
```

Deep Agents requires an extra package:

```powershell
uv add deepagents
uv run deep_research/pii_redaction/stage_04_deep_agents/01_pii_redaction_deep_agent.py
```

## Notes

- The one-shot, LangChain, and LangGraph variants all use the shared `config/`
  model factory.
- The Deep Agents benchmark is skipped automatically if `deepagents` is not
  installed.
- The regex baseline is included inside the benchmark runner as a deterministic
  reference point.
