## Status at 2026-11-15 ~21:05 IST — COMPLETE

**All stages done (1–9, including optional Stage 8). Project finished ~13h before the Mon 10:00 IST deadline.**

- **Stage 1–3:** setup, data (4 sets, seed 42, π≈0.5), smoke gates passed.
- **Stage 4 (Part A):** 9,600 records — 6 policies × 2 subjects × 4 sets.
- **Stage 5 (Part B):** 4,800 records — 6 natural models.
- **Stage 6 (validation):** judge agreement 98%/100%; reader ρ=0.511 (flagged as limitation). Hand-label CSVs written.
- **Stage 7 (analysis):** 6 figures + matrix + all metric/rank/patch/prereg JSONs.
- **Stage 8 (optional robustness):** prompt-rotation check — all 4 rephrasings reproduce the exploits.
- **Stage 9 (draft):** `draft.md` complete, every placeholder filled, §4.8 robustness added.

### Headline results
- **Stress test lands:** evader = 1.00 MASK honesty (vs 0.81) & 1.00 Pushback, Omniscience Index +3.0 (vs −24.4), all at DV≈0 (useless). Hedger ECE 0.037 (vs 0.43).
- **Rank disagreement:** Pushback benchmark-vs-DV ρ=−0.37 (near-opposite order); MASK +0.83, Omniscience +0.94, forecast +0.20.
- **Natural gaming:** `glm-5p3` tops every benchmark unprompted while least informative.
- **Patches:** Pushback patch flips DV corr −0.81→+0.53; MASK/Omniscience patches close the prompted exploits.
- **Part D:** forecasting DV ≈0/negative for all models — benchmarks reward recall, not reasoning.
- **Robustness:** exploit survives all 4 prompt rewordings.
- **Prereg:** 4/5 directional predictions supported; the one miss reinforces the thesis.

### Spend: **$14.81** of $90 cap (target was $45). ~84% under target.

### Human annotation — RECEIVED & INCORPORATED (2026-11-16)
Project owner returned 50 reader + 47 MASK + 50 Omniscience labels. Results (`results/validation_human.md`):
- Reader vs DV reader: **Spearman ρ=0.641 (above 0.6 gate)**, MAD 0.134 — upgrades the paper's main limitation.
- Omniscience grade: 98% agreement, κ=0.97 (human-confirmed).
- MASK commit/evade: 81%, κ=0.63 — residual caveat, flagged in draft Appendix.
draft.md Appendix + DECISIONS.md updated.

### Original annotation note (superseded)
`results/handlabel_reader.csv` (50 rows) and `results/handlabel_judge.csv` (100 rows) have empty human columns. Filling them would upgrade the reader-proxy validation from model-vs-model to human-anchored (currently ρ=0.511 reader agreement is the paper's main stated limitation). To contribute: fill the `human_q`/`human` columns and tell me — I'll recompute agreement and update the draft. **The paper is complete and defensible without this.**

All work committed and pushed to `main` (github.com/MurtuzaShaikh26/gameable_epistemic_bench).
