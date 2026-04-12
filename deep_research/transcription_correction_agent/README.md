# Transcription Correction Agent

A LangGraph agent that automatically detects, corrects, and annotates errors
in ASR (Automatic Speech Recognition) transcripts, with specialised handling
for medical terminology across multiple languages.

## What it does

```
START
  ↓
load ──────────────────────────────────────────────────────────
  ↓
detect_errors ──→ detect_medical
  ↓                    ↓
[Send fan-out]    [Send fan-out]
error_worker(*)   medical_worker(*)   ← run in parallel
  ↓                    ↓
synthesize ←──────────────────────────────────────────────────
  ↓
save_output
  ↓
END
```

### Node responsibilities

| Node | What it does |
|---|---|
| `load` | Loads the JSON transcript + medical glossary |
| `detect_errors` | LLM scans for garbled, incoherent, or repeated segments |
| `detect_medical` | LLM identifies segments with potential medical terminology |
| `error_worker` | Per-segment: extract audio → retranscribe → correct |
| `medical_worker` | Per-segment: extract audio → retranscribe → link to glossary |
| `synthesize` | Merges all CorrectionRecords into the segment list |
| `save_output` | Writes corrected JSON + annotated Markdown |

### Retranscription strategy (per segment)

1. **faster-whisper** — local, offline, fast (requires `pip install faster-whisper`)
2. **multimodal LLM** — OpenAI `gpt-4o-audio-preview` via base64 audio (requires `OPENAI_API_KEY`)
3. **text-only LLM** — always available fallback; uses surrounding context to fix the segment

Set `--method auto` to try them in that order.

## Running

```bash
# Demo with built-in sample transcript (text-only LLM correction)
uv run deep_research/transcription_correction_agent/run.py

# With your own transcript + audio (enables audio retranscription)
uv run deep_research/transcription_correction_agent/run.py \
    --transcript path/to/transcript.json \
    --audio      path/to/audio.mp3 \
    --method     auto
```

## File structure

```
transcription_correction_agent/
├── agent.py              ← LangGraph StateGraph (main logic)
├── state.py              ← Shared TypedDict state schema
├── run.py                ← Entry point & CLI
├── tools/
│   ├── audio_extraction.py   ← ffmpeg-based audio clip extractor
│   ├── retranscription.py    ← faster-whisper / multimodal LLM / text LLM
│   └── medical_terms.py      ← Glossary loader + multilingual term linker
├── data/
│   ├── medical_glossary.json ← 30 medical terms × 8 languages
│   └── sample_transcript.json← Created on first run if absent
└── outputs/              ← Corrected JSON + Markdown written here
```

## Medical glossary

`data/medical_glossary.json` contains 30 medical terms covering:

- Cardiology: myocardial infarction, atrial fibrillation, hypertension, ECG, echocardiogram …
- Oncology: metastasis, carcinoma, lymphoma, chemotherapy …
- Neurology: encephalopathy, epilepsy, neuropathy, dementia …
- Pulmonology: pneumonia, asthma, bronchoscopy …
- Pharmacy: amoxicillin, metformin, lisinopril, atorvastatin, omeprazole …
- Critical care: sepsis …
- Nephrology: hyponatremia, creatinine …

Each term includes:
- Canonical English form
- Translations in ES, FR, DE, PT, AR (romanized), ZH (Pinyin), JA (Romaji)
- ICD-10 code hint
- Common ASR phonetic mishearings (so the LLM can recognize mangled forms)

## Optional dependencies

| Feature | Package | Install |
|---|---|---|
| Audio extraction | ffmpeg (system) | `winget install ffmpeg` or via conda |
| Audio extraction (Python) | pydub | `pip install pydub` |
| Local retranscription | faster-whisper | `pip install faster-whisper` |
| Multimodal retranscription | openai | already in project deps |

If none of the audio tools are available, the agent runs entirely with the
text-only LLM fallback — no audio file is required.

## Output files

`outputs/<transcript-stem>/`

| File | Contents |
|---|---|
| `*.corrected.json` | Full corrected transcript with correction metadata |
| `*.corrected.md` | Annotated readable transcript + multilingual glossary annex |
