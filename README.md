# Saying Nothing Scores Well: A Null-Policy Test for Epistemic Benchmarks

Many epistemic benchmarks reward the *absence* of a bad behaviour (lying, caving to
pushback, miscalibration, hallucination). A model can avoid each by being useless. This
project stress-tests that with degenerate "null policies" and a behaviour-agnostic
**Decision Value (DV)** metric built from a reader model's posterior.

## Reproduce

```bash
pip install -r requirements.txt
cp .env.example .env    # add your FIREWORKS_API_KEY
python tests/test_metrics.py            # Stage 1 gate: unit tests
python src/build_datasets.py            # Stage 2: sample all sets (seed 42)
python src/smoke.py                     # Stage 3: 20-item smoke test + cost
python src/run_partA.py                 # Stage 4: policies x subjects x sets
python src/run_partB.py                 # Stage 5: natural models, baseline
python src/validate.py                  # Stage 6: reader/judge agreement
python src/analyze.py                   # Stage 7: metrics, figures, tables
```

All API responses are cached on disk (`.cache/`) keyed by (model, messages, params),
so no call is ever paid for twice. Spend is tracked in `results/spend.json`.

## Layout
- `src/` — code (metrics, runner, policies, stage scripts)
- `data/samples/` — sampled item IDs (seed 42)
- `results/` — raw outputs, JSONL call logs, spend
- `figures/` — figures
- `draft.md` — the report draft

See `DECISIONS.md` (deviations), `DATA.md` (datasets/licences), `PREREG.md`
(predictions), `STATUS.md` (latest status).

**Note:** All code in this repo was written during the sprint.
