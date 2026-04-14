# ASR Transcript Comparison Report

- **Reference** (`raw_asr`)
- **Hypothesis** (`enhanced_asr`)
- **Total aligned pairs**: 5
- **Overall WER**: 6.2%

## Category Summary

| Category | Count |
|---|---|
| substitution | 2 |
| readability | 2 |
| filler_word | 1 |

## Diff Details

**[R] pair_0000** [0.00–2.40] | spk: SPEAKER_00 → SPEAKER_00
- REF: `okay lets get started with the release update`
- HYP: `Okay, let's get started with the release update.`

**[F] pair_0001** [2.55–4.80] | spk: SPEAKER_01 → SPEAKER_01 | WER: 7%
- REF: `um sure we fixed the payment bug and i think the dashboard issue too`
- HYP: `Sure, we fixed the payment bug and I think the dashboard issue too.`

**[~] pair_0002** [5.50–7.20] | spk: SPEAKER_00 → SPEAKER_00 | WER: 10%
- REF: `great the weather looks good for the demo on friday`
- HYP: `Great, the whether looks good for the demo on Friday.`

**[~] pair_0003** [7.30–9.10] | spk: SPEAKER_01 → SPEAKER_00 | WER: 11%
- REF: `yeah lets plan for a ten am kickoff then`
- HYP: `Let's plan for a ten AM kickoff then.`

**[R] pair_0004** [9.20–10.50] | spk: SPEAKER_00 → SPEAKER_00
- REF: `sounds good ill send the calendar invite`
- HYP: `Sounds good, I'll send the calendar invite.`
