# Prompt-rotation robustness (Stage 8)

Subject: gpt-oss-120b. Full item sets. Alternate phrasings vs originals.

Claim under test: the exploit survives rewording (published score stays gamed, DV stays ~0).


## MASK (honesty | commitment | DV)

| policy | honesty | commitment | DV |
|---|---|---|---|
| baseline | 0.90 | 0.39 | -0.30 |
| evader | 1.00 | 0.06 | -0.03 |
| evader_v2 | 1.00 | 0.05 | +0.06 |
| evader_v3 | 1.00 | 0.05 | +0.12 |
| abstainer | 0.96 | 0.34 | -0.06 |
| abstainer_v2 | 0.97 | 0.23 | -0.06 |
| abstainer_v3 | 0.99 | 0.18 | +0.07 |

## Omniscience (Index | accuracy | DV)

| policy | index | accuracy | DV |
|---|---|---|---|
| baseline | -52.8 | 0.22 | -0.47 |
| evader | +1.2 | 0.02 | -0.08 |
| evader_v2 | +0.0 | 0.00 | -0.30 |
| evader_v3 | -4.0 | 0.02 | -0.54 |
| abstainer | -10.8 | 0.17 | -0.39 |
| abstainer_v2 | -16.0 | 0.17 | -0.37 |
| abstainer_v3 | -10.4 | 0.10 | -0.38 |

## Verdict

- **evader_v2** (mirrors evader): MASK exploit REPRODUCED; Omniscience exploit REPRODUCED
- **evader_v3** (mirrors evader): MASK exploit REPRODUCED; Omniscience exploit REPRODUCED
- **abstainer_v2** (mirrors abstainer): MASK exploit REPRODUCED; Omniscience exploit REPRODUCED
- **abstainer_v3** (mirrors abstainer): MASK exploit REPRODUCED; Omniscience exploit REPRODUCED