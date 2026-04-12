# ASR Medical Verification Agent

A comprehensive Deep Agent system for transcription error detection, verification, and correction with specialized medical terminology support.

## Overview

This agent system provides a complete pipeline for:

1. **Error Detection** - Identify potential errors in ASR transcriptions (low confidence words, homophones, unknown terms)
2. **Audio Extraction** - Extract corresponding audio segments for re-verification
3. **Transcription Verification** - Verify using:
   - Faster-Whisper re-transcription
   - Multimodal LLM (GPT-4o, Claude, etc.)
4. **Medical Term Detection** - Identify medical terms using a multilingual glossary
5. **Correction Application** - Update transcription with verified corrections

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    TRANSCRIPTION VERIFICATION AGENT                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐ │
│  │  Stage 1:       │  │  Stage 2:       │  │  Stage 3:               │ │
│  │  Error          │──▶│  Audio          │──▶│  Verification           │ │
│  │  Detection      │  │  Extraction     │  │  (Whisper/LLM)          │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────┘ │
│          │                      │                      │               │
│          ▼                      ▼                      ▼               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐ │
│  │  Detect:        │  │  Extract:       │  │  Verify:                │ │
│  │  - Low conf     │  │  - Error        │  │  - Re-transcribe        │ │
│  │  - Homophones   │  │    segments     │  │  - Multimodal LLM       │ │
│  │  - Unknown      │  │  - Medical      │  │  - Compare results      │ │
│  │    terms        │  │    term audio   │  │                         │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────┘ │
│                                                                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐ │
│  │  Stage 4:       │  │  Stage 5:       │  │  Output:                │ │
│  │  Medical Terms  │──▶│  Deep Agent     │──▶│  - Corrected            │ │
│  │                 │  │  Orchestration  │  │    transcript           │ │
│  └─────────────────┘  └─────────────────┘  │  - Verification         │ │
│                                            │    report               │ │
│  Medical Term Features:                    │  - Medical term         │ │
│  - Multilingual glossary                   │    analysis             │ │
│  - Cross-language linking                  └─────────────────────────┘ │
│  - Context-aware detection                                            │ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

```bash
# Install faster-whisper for audio transcription
pip install faster-whisper

# Ensure ffmpeg is installed (for audio extraction)
# macOS: brew install ffmpeg
# Ubuntu: sudo apt-get install ffmpeg
# Windows: choco install ffmpeg
```

### Basic Usage

```python
from deep_research.asr_medical_verification import run_verification

# Run complete verification pipeline
report = run_verification(
    transcript_path="path/to/transcript.json",
    audio_path="path/to/audio.wav",  # Optional but recommended
    output_dir="./verification_output",
)

print(f"Errors found: {report['summary']['errors_found']}")
print(f"Medical terms: {report['summary']['medical_terms_found']}")
print(f"Corrections applied: {report['summary']['corrections_applied']}")
```

### Individual Stages

You can also run individual stages:

```bash
# Stage 1: Error Detection
uv run deep_research/asr_medical_verification/stage_01_error_detection/01_identify_errors.py

# Stage 2: Audio Extraction (requires audio file)
uv run deep_research/asr_medical_verification/stage_02_audio_extraction/01_extract_audio_segments.py

# Stage 3a: Whisper Re-transcription
uv run deep_research/asr_medical_verification/stage_03_transcription_verification/01_whisper_retranscribe.py

# Stage 3b: Multimodal LLM Verification
uv run deep_research/asr_medical_verification/stage_03_transcription_verification/02_multimodal_llm_verify.py

# Stage 4: Medical Term Detection
uv run deep_research/asr_medical_verification/stage_04_medical_terms/01_medical_term_detection.py

# Stage 5: Deep Agent (complete workflow)
uv run deep_research/asr_medical_verification/stage_05_deep_agent/01_transcription_verification_agent.py
```

## Features

### 1. Error Detection

- **Low Confidence Detection**: Flags words with ASR confidence < 0.75
- **Homophone Detection**: Identifies potentially confused words (their/there/they're, etc.)
- **Unknown Term Detection**: Finds words that don't match known dictionaries
- **Context Preservation**: Captures surrounding text for better correction

### 2. Audio Extraction

- **Precise Extraction**: Extract exact audio segments for flagged words
- **Configurable Padding**: Add context before/after for better verification
- **Multiple Formats**: Support for WAV, MP3, FLAC output
- **Batch Processing**: Extract multiple segments efficiently

### 3. Transcription Verification

#### Faster-Whisper
- Local, fast re-transcription
- Multiple model sizes (tiny to large-v3)
- Word-level timestamps
- Language auto-detection

#### Multimodal LLM
- Use GPT-4o, Claude, or other multimodal models
- Context-aware verification
- Medical terminology expertise
- Human-like reasoning about audio

### 4. Medical Term Detection

#### Multilingual Glossary
Pre-loaded with medical terms in multiple languages:
- English (en)
- Spanish (es)
- French (fr)
- German (de)
- Italian (it)
- Portuguese (pt)
- Chinese (zh)
- Japanese (ja)
- Arabic (ar)
- Hindi (hi)

#### Categories
- Anatomy (heart, lung, brain, etc.)
- Diseases (diabetes, hypertension, pneumonia, etc.)
- Procedures (surgery, biopsy, MRI, CT, etc.)
- Medications (aspirin, antibiotics, insulin, etc.)
- Vital Signs (blood pressure, heart rate, O2 saturation, etc.)

#### Cross-Language Linking
Each term is linked across languages, allowing the LLM to:
- Recognize medical terms in different languages
- Understand context even when spoken in non-English
- Apply medical knowledge across language barriers

### 5. Deep Agent Orchestration

The Deep Agent provides:
- **Task Planning**: Automatically plans the verification workflow
- **Subagent Delegation**: Can spawn specialized agents for different tasks
- **Memory**: Remembers context across the verification session
- **Human-in-the-Loop**: Can pause for human approval on critical corrections
- **File Management**: Handles reading/writing of transcripts and reports

## Project Structure

```
asr_medical_verification/
├── __init__.py
├── README.md
├── shared/
│   ├── __init__.py
│   ├── transcript_utils.py      # Transcript data models and utilities
│   ├── medical_glossary.py      # Multilingual medical glossary
│   └── audio_utils.py           # Audio extraction and transcription
├── stage_01_error_detection/
│   ├── __init__.py
│   └── 01_identify_errors.py    # Error detection
├── stage_02_audio_extraction/
│   ├── __init__.py
│   └── 01_extract_audio_segments.py  # Audio segment extraction
├── stage_03_transcription_verification/
│   ├── __init__.py
│   ├── 01_whisper_retranscribe.py    # Faster-Whisper verification
│   └── 02_multimodal_llm_verify.py   # Multimodal LLM verification
├── stage_04_medical_terms/
│   ├── __init__.py
│   └── 01_medical_term_detection.py  # Medical term detection
└── stage_05_deep_agent/
    ├── __init__.py
    └── 01_transcription_verification_agent.py  # Main Deep Agent
```

## API Reference

### Transcript Utilities

```python
from deep_research.asr_medical_verification import load_transcript, save_document

# Load a transcript (WhisperX JSON format)
doc = load_transcript("transcript.json", audio_path="audio.wav")

# Access segments
for segment in doc.segments:
    print(f"[{segment.start:.2f}-{segment.end:.2f}] {segment.speaker}: {segment.text}")
    for word in segment.words:
        print(f"  {word.text} (confidence: {word.score:.2f})")

# Save corrected transcript
save_document(doc, "corrected.json")
```

### Medical Glossary

```python
from deep_research.asr_medical_verification import load_medical_glossary, MedicalTerm

# Load default glossary
glossary = load_medical_glossary()

# Search for terms
results = glossary.search("heart", max_results=5)

# Detect terms in text
text = "The patient has hypertension and needs aspirin"
matches = glossary.detect_in_text(text)
for matched_text, term in matches:
    print(f"Found: {matched_text} -> {term.english} ({term.category})")

# Get translations
corazon = glossary.get_term("anatomy_heart")
print(f"English: {corazon.english}")
print(f"Spanish: {corazon.get_term('es')}")
print(f"Chinese: {corazon.get_term('zh')}")

# Add custom terms
custom_term = MedicalTerm(
    term_id="custom_001",
    english="my custom term",
    category="procedure",
    translations={"es": "mi término personalizado"},
)
glossary.add_term(custom_term)
```

### Audio Extraction

```python
from deep_research.asr_medical_verification import AudioSegmentExtractor

# Initialize extractor
extractor = AudioSegmentExtractor("audio.wav")

# Get duration
duration = extractor.get_duration()

# Extract a segment (with padding)
segment = extractor.extract_segment(
    start_time=10.5,
    end_time=15.0,
    output_format="wav",
    sample_rate=16000,
    padding=1.0,  # 1 second before and after
)

# Save to file
segment.save("extracted_segment.wav")

# Or use as bytes
audio_data = segment.audio_data
```

### Faster-Whisper Transcription

```python
from deep_research.asr_medical_verification.shared.audio_utils import FasterWhisperTranscriber

# Initialize transcriber
transcriber = FasterWhisperTranscriber(model_size="base")

# Transcribe audio
result = transcriber.transcribe("audio.wav", language="en")

print(f"Text: {result['text']}")
print(f"Language: {result['language']}")
for seg in result['segments']:
    print(f"[{seg['start']:.2f}-{seg['end']:.2f}] {seg['text']}")
```

## Configuration

### Environment Variables

```bash
# LLM Provider (for multimodal verification)
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here

# Or use local models via Ollama (default)
OLLAMA_MODEL=gemma4:e2b
```

### Custom Glossary

You can load a custom medical glossary:

```python
# Load from file
glossary = load_medical_glossary("my_glossary.json")

# Or create and save
glossary = MultilingualMedicalGlossary()
glossary.add_term(my_custom_term)
glossary.save("my_glossary.json")
```

Glossary JSON format:

```json
{
  "terms": [
    {
      "term_id": "anatomy_heart",
      "english": "heart",
      "category": "anatomy",
      "translations": {
        "es": "corazón",
        "fr": "cœur"
      },
      "synonyms": ["cardiac", "myocardium"],
      "abbreviations": ["H"],
      "related_terms": ["anatomy_cardiac_cycle"]
    }
  ]
}
```

## Advanced Usage

### Custom Error Detection

```python
from deep_research.asr_medical_verification.shared.transcript_utils import (
    load_transcript,
    find_low_confidence_regions,
)

doc = load_transcript("transcript.json")

# Custom confidence threshold
candidates = find_low_confidence_regions(doc, threshold=0.6, context_words=5)

for c in candidates:
    print(f"[{c.score:.2f}] '{c.word}' in context: '{c.left_context} [...] {c.right_context}'")
```

### Batch Processing

```python
from pathlib import Path
from deep_research.asr_medical_verification import run_verification

# Process multiple files
input_dir = Path("./transcripts")
for transcript_file in input_dir.glob("*.json"):
    audio_file = transcript_file.with_suffix(".wav")
    if audio_file.exists():
        report = run_verification(
            transcript_path=str(transcript_file),
            audio_path=str(audio_file),
            output_dir=f"./output/{transcript_file.stem}",
        )
        print(f"Processed {transcript_file.name}")
```

### Integration with Deep Agents

```python
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from deep_research.asr_medical_verification.stage_05_deep_agent._01_transcription_verification_agent import (
    create_verification_tools,
)

# Create custom agent with verification tools
tools = create_verification_tools()

agent = create_deep_agent(
    model="claude-sonnet-4-5-20250929",
    tools=tools,
    system_prompt="You are a medical transcription specialist...",
    backend=FilesystemBackend(root_dir=".", virtual_mode=True),
)

# Use the agent
result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "Verify this medical transcription and identify any errors"
    }]
}, config={"configurable": {"thread_id": "session-1"}})
```

## Limitations & Future Work

### Current Limitations

1. **Audio Format**: Requires ffmpeg for audio extraction
2. **Whisper Model**: Faster-Whisper requires local model download
3. **Multimodal LLM**: Requires API access to GPT-4o or similar
4. **Medical Glossary**: Default glossary is basic; domain-specific terms need to be added

### Planned Improvements

1. **Enhanced Glossary**: Expand to include more specialized medical fields
2. **Active Learning**: Learn from corrections to improve detection
3. **Speaker Profiles**: Track speaker-specific patterns and terminology
4. **Integration**: Direct integration with WhisperX for end-to-end pipeline
5. **Streaming**: Real-time error detection and correction

## Troubleshooting

### ffmpeg not found

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# Windows
choco install ffmpeg
```

### faster-whisper not installed

```bash
pip install faster-whisper
```

### No audio path in transcript

The transcript JSON should have an `audio_path` field, or you can provide it explicitly:

```python
doc = load_transcript("transcript.json", audio_path="audio.wav")
```

## License

This project is part of the test-langchain educational repository.

## Contributing

This is an educational project. Feel free to extend the glossary, add new error detection strategies, or improve the verification methods.
