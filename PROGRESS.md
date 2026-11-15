# PROGRESS.md

- 2026-11-15 18:11 IST — Stage 1 (setup) START.
- 2026-11-15 18:11 IST — Stage 1 gate PASSED: 12/12 metric unit tests pass (DV, Brier, market proxy, ECE, Brier skill, Omniscience index, Kalai, Spearman, bootstrap). Runner with cache/backoff/spend written. Models queried and mapped (see DECISIONS.md).
- 2026-11-15 18:30 IST — Stage 2 (data) DONE. All 4 sets built. MASK 200 (pi=0.500), Pushback 200 (pi=0.500), Omniscience 250 (pi=0.500), ForecastBench 150 (pi=0.447). Gate PASSED (4/4, each >=150). ForecastBench fetched via jsDelivr (raw.githubusercontent reset).
- 2026-11-15 ~19:45 IST — Stage 3 (smoke) DONE/gate PASSED. Cost ~$25 full-run est (<$45). q-parse 2.4% OK; reader sanity OK (3 main sets; forecast +0.18 borderline, Part D secondary). Compliance: evader 97%, hedger 84%, stonewaller 98%; abstainer 74% & yes_man(control) accepted as limitations. Launching Stage 4 (Part A full).
