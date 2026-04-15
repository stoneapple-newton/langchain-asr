# ASR Transcript Comparison Report

- **Reference** (`whisper`)
- **Hypothesis** (`azure`)
- **Total aligned pairs**: 9
- **Overall WER**: 26.0%

## Category Summary

| Category | Count |
|---|---|
| substitution | 4 |
| insertion | 1 |
| deletion | 1 |
| readability | 1 |
| filler_word | 2 |

## Diff Details

**[R] pair_0000** [0.00–3.20] | spk: SPEAKER_00 → SPEAKER_00
- REF: `alright everyone lets kick off the product review for q3`
- HYP: `Alright everyone, let's kick off the product review for Q3.`

**[F] pair_0001** [3.40–6.10] | spk: SPEAKER_01 → SPEAKER_01 | WER: 10%
- REF: `uh thanks we shipped the new onboarding flow last week`
- HYP: `Thanks, we shipped the new onboarding flow last week.`

**[~] pair_0002** [6.20–9.00] | spk: SPEAKER_01 → SPEAKER_01 | WER: 10%
- REF: `conversion rate went up buy twelve percent since the launch`
- HYP: `Conversion rate went up by twelve percent since the launch.`

**[~] pair_0003** [9.10–11.50] | spk: SPEAKER_00 → SPEAKER_00 | WER: 18%
- REF: `thats great what about the drop off on the payment page`
- HYP: `That's great. What about the drop-off on the payment page?`

**[F] pair_0004** [11.60–13.80] | spk: SPEAKER_02 → SPEAKER_01 | WER: 10%
- REF: `um still about thirty percent of users abandon at checkout`
- HYP: `Still about thirty percent of users abandon at checkout.`

**[-] pair_0005** [13.81–15.20] | spk: SPEAKER_02 → None
- REF: `checkout`
- HYP: `None`

**[~] pair_0006** [15.30–18.50] | spk: SPEAKER_00 → SPEAKER_00 | WER: 9%
- REF: `we need to address that before the end of quarter two`
- HYP: `We need to address that before the end of quarter four.`

**[~] pair_0007** [18.60–21.00] | spk: SPEAKER_01 → SPEAKER_02 | WER: 14%
- REF: `agreed ill set up an a b test on the checkout flow next sprint`
- HYP: `Agreed, I'll set up an A/B test on the checkout flow next sprint.`

**[+] ins_0007** [21.60–24.00] | spk: None → SPEAKER_00
- REF: ``
- HYP: `Perfect. Let's also review the mobile crash reports from last night.`
