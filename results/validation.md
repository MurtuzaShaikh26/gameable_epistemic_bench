# Validation (Stage 6)

Reader1=qwen3p8-max vs Reader2=kimi-k3 on 50 items:
- mean |q1-q2| = 0.165
- Spearman ρ = 0.511  (gate: >=0.6)

Judge1=qwen3p8-max vs Judge2=minimax-m3:
- MASK commit agreement = 98.0% (n=50)
- Omniscience grade agreement = 100.0% (n=50)

Hand-label CSVs written with empty human columns (non-blocking): results/handlabel_reader.csv, results/handlabel_judge.csv
