# ASR Quality Agent Catalog

This catalog collects concrete agent ideas for improving automatic speech
recognition quality across `deep_research/asr` and `deep_research/asr-v2`.

The design is grounded in current ASR pain points documented by cloud speech
providers, WhisperX, and recent research:

- diarization and speaker attribution
- multi-channel routing
- domain vocabulary and phrase boosting
- automatic punctuation and readability cleanup
- code-switching and multilingual evaluation
- confidence-driven error detection
- impact-aware review for sensitive domains
- PHI / PII detection and redaction

Pragmatic note: many of these are better implemented as deterministic nodes or
guarded LangGraph workers than as unconstrained free-form agents. "Agent" here
means a focused quality-improvement role with its own inputs, policy, and
output contract.

## Suggested Agents

### 1. Audio Intake Agent

- Goal: reject or flag low-quality audio before transcription.
- Watches for: low sample rate, clipping, long silence, channel mismatch, bad
  duration metadata, unexpected codecs.
- Inputs: raw media metadata, ffmpeg probe output, waveform stats.
- Outputs: normalized ingest report plus recommended preprocessing actions.
- Best fit: deterministic tool + rule engine.

### 2. VAD Segmentation Agent

- Goal: improve chunk boundaries before ASR and reduce hallucinated text in
  silence.
- Watches for: overlong segments, speech-in-noise regions, silence stretches,
  clipped speech starts and ends.
- Inputs: waveform, VAD scores, initial segment timing.
- Outputs: revised speech regions with confidence labels.
- Best fit: deterministic preprocessing node.

### 3. Multi-Channel Routing Agent

- Goal: exploit separate channels when audio already isolates speakers.
- Watches for: call-center stereo, dual-mic recordings, meeting platform stems.
- Inputs: channel count, per-channel RMS, transcript/channel tags.
- Outputs: channel-aware transcript plan or fallback to diarization.
- Why it matters: Google and AWS both document channel-aware transcription as a
  separate path from generic diarization.

### 4. Speaker Count Estimator Agent

- Goal: choose a realistic speaker range before diarization.
- Watches for: alternating turn patterns, embedding clusters, overlap density.
- Inputs: speaker embeddings, segment timings, lexical turn structure.
- Outputs: min/max speaker estimate with rationale.
- Best fit: deterministic heuristics with optional embedding clustering.

### 5. Diarization Repair Agent

- Goal: fix missing or unstable speaker labels after first-pass diarization.
- Watches for: `UNKNOWN` speakers, impossible speaker flips, one-word turns,
  identity drift across nearby segments.
- Inputs: transcript JSON, word timings, speaker IDs, local context windows.
- Outputs: corrected speaker labels and a change log.
- Repo fit: this maps directly to `stage_03_diarization`.

### 6. Overlap Resolution Agent

- Goal: handle overlapping speech explicitly instead of flattening it into one
  line.
- Watches for: interleaved words, timestamp collisions, dense back-and-forth,
  simultaneous acknowledgements.
- Inputs: word-level timestamps, channel info if available, diarization output.
- Outputs: overlap annotations, split segments, or "manual review" flags.
- Why it matters: diarization outputs often linearize overlap poorly.

### 7. Backchannel Merge Agent

- Goal: merge low-information acknowledgements into nearby turns when they are
  not semantically important.
- Watches for: "yeah", "right", "mm-hmm", "okay" turns shorter than a timing
  threshold.
- Inputs: segment text, duration, speaker continuity, pause length.
- Outputs: merged transcript turns plus audit trail.
- Repo fit: useful in both `asr` and `asr-v2` cleanup graphs.

### 8. Timestamp Alignment Audit Agent

- Goal: detect when lexical content and timing disagree after forced alignment.
- Watches for: negative durations, non-monotonic word order, long text packed
  into short spans, punctuation attached to the wrong speaker boundary.
- Inputs: aligned words, segment boundaries, subtitle exports.
- Outputs: alignment anomalies and recommended fixes or re-alignment.
- Why it matters: WhisperX exists largely because raw Whisper timestamps can be
  coarse at the word level.

### 9. Confidence Triage Agent

- Goal: focus expensive correction work on the most error-prone spans.
- Watches for: low word confidence, dense low-confidence clusters, unusual
  token sequences, disagreement between model alternatives.
- Inputs: word confidence scores, transcript text, optional N-best hypotheses.
- Outputs: ranked review queue for downstream correction agents.
- Research signal: recent ASR error detection work shows confidence features are
  useful, not just transcript text.

### 10. Phonetic Error Correction Agent

- Goal: repair acoustically plausible substitutions.
- Watches for: homophones, near-homophones, acronym confusions, phonetic miss
  matches such as "jwt" vs "j w t" or brand-name distortions.
- Inputs: transcript span, confidence scores, pronunciation lexicon, domain
  vocabulary.
- Outputs: candidate corrections with confidence and evidence.
- Best fit: LLM or seq2seq corrector gated by phonetic and confidence signals.

### 11. Domain Vocabulary Agent

- Goal: protect domain-specific terms, acronyms, proper nouns, and product
  names.
- Watches for: rare terms, organization names, jargon, recurring misheard
  tokens.
- Inputs: transcript, project glossary, prior corrections, customer or meeting
  context.
- Outputs: phrase boosts, correction dictionary, term-normalization patches.
- Why it matters: AWS, Google, and Azure all expose custom vocabulary or phrase
  list features for this exact problem.

### 12. Readability Editor Agent

- Goal: improve punctuation, casing, spacing, and light phrasing without
  changing facts.
- Watches for: run-on lines, missing capitals, repeated fillers, malformed
  contractions, subtitle-hostile line breaks.
- Inputs: cleaned transcript segments plus strict formatting rules.
- Outputs: same-line-count revised transcript.
- Repo fit: this already exists in `asr-v2`; the role can be expanded into a
  reusable worker pattern.

### 13. Code-Switch Detection Agent

- Goal: mark language boundaries inside a single utterance.
- Watches for: token-level language switches, transliteration, borrowed words,
  mixed scripts.
- Inputs: transcript text, language ID scores, token script features.
- Outputs: token or span-level language tags and confidence.
- Why it matters: recent ACL work continues to treat code-switching as a hard
  ASR setting, especially for short or low-resource mixed-language segments.

### 14. Code-Switch Rewrite Agent

- Goal: repair mixed-language spans without forcing them into one language.
- Watches for: mistranscribed foreign terms, script normalization issues,
  inconsistent transliteration.
- Inputs: code-switch tags, bilingual glossary, surrounding discourse context.
- Outputs: corrected mixed-language transcript plus alternate renderings where
  useful.
- Evaluation note: ordinary WER can understate or mis-score these cases.

### 15. Context Grounding Agent

- Goal: use retrieved context to resolve ambiguous technical references.
- Watches for: pronouns without antecedents, ambiguous acronym expansions,
  project names, deadlines, participant names.
- Inputs: transcript span, meeting docs, project notes, participant roster.
- Outputs: grounded correction suggestions with citations to retrieved context.
- Repo fit: this is the natural extension of `stage_05_rag_context`.

### 16. Severity Review Agent

- Goal: prioritize errors by downstream harm, not just token mismatch count.
- Watches for: medication names, numbers, dates, negation, commitments, legal
  or financial terms, patient-facing or safety-critical content.
- Inputs: transcript diffs, domain policy, extracted entities.
- Outputs: severity labels such as low, medium, high, manual-review-required.
- Research signal: recent work argues WER alone misses downstream risk.

### 17. Compliance Redaction Agent

- Goal: detect and redact PHI / PII after transcript stabilization.
- Watches for: names, addresses, dates, account numbers, medical identifiers,
  sensitive policy triggers.
- Inputs: transcript, entity recognizer output, domain rules.
- Outputs: redacted transcript plus structured entity ledger.
- Why it matters: AWS and Azure document domain-sensitive speech workflows
  where privacy handling matters as much as raw accuracy.

### 18. Human Escalation Agent

- Goal: route only the hardest transcript spans to a human reviewer.
- Watches for: unresolved disagreement between agents, low-confidence critical
  entities, overlap-heavy spans, repeated repair failures.
- Inputs: confidence triage, severity labels, correction diffs.
- Outputs: compact review packets with evidence and recommended decision.
- Best fit: LangGraph human-in-the-loop node, not a free-running agent.

## Recommended Build Order In This Repo

If you want to turn this catalog into concrete repo examples, the highest-value
next additions are:

1. `Confidence Triage Agent`
2. `Domain Vocabulary Agent`
3. `Overlap Resolution Agent`
4. `Severity Review Agent`
5. `Compliance Redaction Agent`

That order matches the current repo progression well:

- `asr` already has diarization, RAG grounding, and production orchestration.
- `asr-v2` already has deterministic cleanup and guarded readability editing.
- The biggest missing gaps are confidence-aware routing, overlap handling,
  glossary protection, and risk-aware review.

## Source Notes

These agent ideas were inferred from the current repo plus these external
sources:

- Google Cloud Speech-to-Text diarization:
  https://docs.cloud.google.com/speech-to-text/v2/docs/multiple-voices
- Google Cloud Speech-to-Text multi-channel transcription:
  https://docs.cloud.google.com/speech-to-text/docs/multi-channel
- Google Cloud Speech-to-Text model adaptation:
  https://docs.cloud.google.com/speech-to-text/docs/adaptation
- Google Cloud automatic punctuation:
  https://docs.cloud.google.com/speech-to-text/docs/v1/automatic-punctuation
- Amazon Transcribe diarization:
  https://docs.aws.amazon.com/transcribe/latest/dg/diarization.html
- Amazon Transcribe multi-channel audio:
  https://docs.aws.amazon.com/transcribe/latest/dg/channel-id.html
- Amazon Transcribe custom vocabularies:
  https://docs.aws.amazon.com/transcribe/latest/dg/custom-vocabulary.html
- Amazon Transcribe accuracy customization:
  https://docs.aws.amazon.com/transcribe/latest/dg/improving-accuracy.html
- Amazon Transcribe PHI identification:
  https://docs.aws.amazon.com/transcribe/latest/dg/phi-id.html
- Azure Speech overview:
  https://learn.microsoft.com/en-us/azure/ai-services/speech-service/overview?tabs=curl
- Azure diarization quickstart:
  https://learn.microsoft.com/en-us/azure/ai-services/speech-service/get-started-stt-diarization
- Azure custom speech overview:
  https://learn.microsoft.com/en-us/azure/ai-services/speech-service/custom-speech-overview
- WhisperX project page:
  https://github.com/m-bain/whisperX
- RED-ACE confidence-aware ASR error detection:
  https://aclanthology.org/2022.emnlp-main.180/
- Code-switching ASR language-ID guidance:
  https://aclanthology.org/2023.calcs-1.4/
- PolyWER for code-switched evaluation:
  https://aclanthology.org/2024.findings-emnlp.356/
- Severity-aware ASR evaluation:
  https://aclanthology.org/2023.bionlp-1.6/
- Clinical-risk critique of WER:
  https://aclanthology.org/2026.iwsds-1.39/
