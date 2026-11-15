# Human annotation validation (Stage 6b)

Human labels returned by the project owner, compared to the model reader/judge used in the main results. These are the human-anchored numbers the paper's reader-proxy limitation was waiting on.

## Reader (human q vs DV reader q), n=50
- mean |q_model - q_human| = 0.134
- Pearson r = 0.687;  Spearman rho = 0.641
- 3-way directional agreement (false / uninformative / true lean) = 76%

## MASK commit-vs-evade judge (human vs model), n=47 (2 blank rows dropped)
- raw agreement = 80.9%;  Cohen kappa = 0.63
- disagreements concentrate on refusal-with-explanation answers (model scores a stated fact as COMMIT; human reads the refusal framing as EVADE).

## AA-Omniscience grade judge (human vs model), n=50
- raw agreement = 98.0%;  Cohen kappa = 0.97
