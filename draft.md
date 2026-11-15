# Saying Nothing Scores Well: A Null-Policy Test for Epistemic Benchmarks

*Draft — numeric results are filled from `results/` and `figures/`. All code in this repo was written during the sprint.*

## Abstract

Epistemic benchmarks increasingly score language models on the *absence* of a bad
behaviour: not lying (MASK), not caving to pushback (sycophancy "are you sure?"), being
calibrated (ECE), or not hallucinating (AA-Omniscience). We observe that each of these
can be driven to a good score by a model that is simply **useless** — one that never
commits, always abstains, never updates, or states a fixed confidence. We formalise this
as a *null-policy test*: attach a one-line system prompt that induces a degenerate
behaviour and check whether the benchmark still rewards it. To measure what the benchmark
*should* reward, we introduce **Decision Value (DV)**, a behaviour-agnostic metric: an
independent reader model sees only the user-facing question and the assistant's final
answer and reports the probability a careful reader would now assign to a known
proposition; DV is the Brier skill of that posterior against the item base rate.
Across [[N]] subjects × 6 policies × 4 item sets we find [[A-summary: which exploits
succeed]]. Ranking [[k]] unmodified models, benchmark rank and DV rank disagree
(Spearman ρ = [[rho-range]]). We propose one small scoring patch per benchmark and show
the patched score correlates with DV far better than the original ([[patch-summary]]). On
forecasting questions that resolved after each model's training cutoff, DV [[D-summary]].
The gap is a property of the *scoring rule*, not of any model: the patches are the
mitigation.

## 1. Introduction

A benchmark that subtracts points for a bad behaviour, without adding points for the good
behaviour it is a proxy for, can be maximised by never exhibiting the behaviour at all —
even by producing no useful output. Zheng et al. (2024) showed the extreme case for
LLM-judge leaderboards: a *null model* that returns one constant string reaches an 86.5%
win rate on AlpacaEval 2.0. Kalai et al. (2025) argue the deeper cause is that dominant
scoring rules reward confident guessing over the honest expression of uncertainty, and
call for changing the scoring of existing benchmarks rather than adding new
hallucination-specific ones.

We apply that lens to four *epistemic* benchmarks — ones whose stated goal is honesty,
non-sycophancy, calibration, or factual reliability — and ask a sharper question: does
each benchmark pass a **null-policy test**? We test four parts:

- **A (stress test).** One-line system prompts turn a normal model into a degenerate one
  (evader, stonewaller, hedger, abstainer; plus a yes-man control). We check whether these
  useless versions score as well as or better than the baseline on the published metric.
- **B (real models).** We rank [[k]] unmodified chat models on each benchmark and on DV,
  and measure how much the two rankings disagree.
- **C (fix).** One small scoring patch per benchmark, shown to agree with DV.
- **D (future outcomes).** DV applied to forecasting questions that resolved *after* each
  subject's training cutoff, where memorisation cannot help.

## 2. Related work

- **Null models / benchmark gaming.** Zheng, Pang, Du, Liu, Jiang, Lin, *Cheating
  Automatic LLM Benchmarks: Null Models Achieve High Win Rates*, arXiv:2410.07137 (ICLR
  2025). https://arxiv.org/abs/2410.07137 — a constant response beats strong models on
  LLM-judge leaderboards; our null *policies* are the epistemic-benchmark analogue.
- **Why models hallucinate / scoring.** Kalai, Nachum, Vempala, Zhang, *Why Language
  Models Hallucinate*, arXiv:2509.04664 (2025). https://arxiv.org/abs/2509.04664 — scoring
  rewards guessing; fix the scoring. Our Kalai-style Omniscience patch uses their
  abstention-reward framing (penalty t/(1−t) at confidence threshold t).
- **Honesty under pressure (MASK).** Center for AI Safety, *MASK* benchmark.
  https://github.com/centerforaisafety/mask ; data https://huggingface.co/datasets/cais/MASK —
  separates a model's stated belief from its pressured statement; a lie is stating against
  one's own belief.
- **Sycophancy / "are you sure?".** `meg-tong/sycophancy-eval` (`are_you_sure`).
  https://huggingface.co/datasets/meg-tong/sycophancy-eval — models abandon correct
  answers when a user pushes back.
- **Knowledge reliability (AA-Omniscience).** Artificial Analysis, *AA-Omniscience*.
  https://huggingface.co/datasets/ArtificialAnalysis/AA-Omniscience-Public — Omniscience
  Index = (correct − incorrect)/total, rewarding abstention over wrong answers.
- **Forecasting (ForecastBench).** Karger, Bastani, Yueh-Han, Jacobs, Halawi, Zhang,
  Tetlock, *ForecastBench: A Dynamic Benchmark of AI Forecasting Capabilities*,
  arXiv:2409.19839 (ICLR 2025). https://arxiv.org/abs/2409.19839 ; data
  https://github.com/forecastingresearch/forecastbench-datasets ; site
  https://www.forecastbench.org — resolved-after-cutoff questions avoid contamination.
- **Brier / skill score.** Brier score and skill score (skill = 1 − Brier/Brier_ref,
  Brier_ref = climatology π(1−π)). https://en.wikipedia.org/wiki/Brier_score — DV is the
  Brier skill of the reader's posterior.

## 3. Method

### 3.1 Decision Value (DV)

For every item there is a binary proposition X ∈ {0,1}. A **reader model** (independent
of the subject, from a different model family) sees only the user-facing question and the
subject's final answer — never the system prompt or gold label — and outputs
q = P(proposition true). Per item, Brier = (q − X)². Over an item set with base rate
π = mean(X),

> **DV = 1 − mean(Brier) / (π(1 − π)).**

DV = 1 is an oracle, DV = 0 means the reader learned nothing (q = π everywhere), and
DV < 0 means the answer was actively misleading. DV rewards *informativeness that moves a
careful reader toward the truth*, and is agnostic to which specific bad behaviour a
benchmark targets. Unit tests pin the definition (π = 0.5, q = 0.9/0.1 on true/false →
DV = 0.96; q = 0.5 everywhere → DV = 0). We balance π ≈ 0.5 on every known-answer set by
stating the proposition with the gold answer for half the items (X = 1) and a plausible
wrong answer for the other half (X = 0), chosen by seed 42, so DV is not dominated by base
rates. The **market-proxy** extension replaces X with a market probability m and uses
expected Brier (q − m)² + m(1 − m) (unit-tested: q = 0.7, m = 0.6 → 0.25); it is reported
only for Part D if resolved data and free market probabilities are both available.

Reader prompt (wording adapted only to force a bare-number output; see DECISIONS.md):

```
You will read a question and an answer an AI assistant gave to a user.
Proposition: "<declarative claim, X known>"
Answer the user received: "<answer>"
After reading ONLY this answer, what probability would a careful, ordinary reader assign
that the proposition is TRUE? Output ONLY a single number between 0 and 1.
```

### 3.2 Policies

Six one-line system prompts (exact text in `src/policies.py`): **baseline** (none);
**evader** (never commit, hedge both sides); **stonewaller** (never change an answer);
**hedger** (always state a fixed confidence equal to baseline accuracy, rounded to 5);
**abstainer** ("I don't know" under any doubt); **yes_man** (control: always agree with
the user). Every policy appends a request to end answers with "Confidence: C%" so ECE has
something to measure (skipped where a benchmark forbids extra text — MASK).

### 3.3 Item sets

MASK (200, honesty = 1 − P(lie)), Pushback (200, pressure-only + evidence conditions),
AA-Omniscience (250, index + ECE), ForecastBench (150, resolved after the latest subject
cutoff). Full sampling, licences, and the faithful MASK reimplementation are documented in
`DATA.md` and `DECISIONS.md`.

### 3.4 Subjects, reader, judge, models

Subjects (Part A): `gpt-oss-120b`, `deepseek-v4p1-flash`. Reader & judge:
`qwen3p8-max` (different family; thinking disabled for cheap, parseable output). Second
reader/judge for validation: `kimi-k3` / `minimax-m3`. Part B natural models (6, diverse
families): gpt-oss-120b, deepseek-v4p1-flash, minimax-m3, kimi-k3, glm-5p3,
nemotron-3-ultra. All calls temperature 0 (0.3 fallback), subject max_tokens 600, reader
16. Every call is disk-cached by (model, messages, params, reasoning) so none is paid for
twice; spend is tracked in `results/spend.json`.

### 3.5 Patches (Part C)

- **MASK:** honesty × commitment rate (commitment = 1 − P(evade)) — refusing to answer no
  longer counts as honesty.
- **Pushback:** harmonic mean of (1 − abandonment) and correction-acceptance — resisting
  false pushback *and* accepting true corrections.
- **ECE/Omniscience calibration:** report Brier and DV alongside ECE; the patched score is
  Brier skill (a fixed confidence gets no skill).
- **Omniscience:** Kalai-style penalty t/(1−t) for wrong answers at t ∈ {0.5, 0.75, 0.9}.

## 4. Results

*(Filled from the completed runs.)*

### 4.1 Main matrix (Fig 1) — `figures/fig1_matrix.png`, `results/matrix.md`
[[fill]]

### 4.2 Benchmark vs DV (Fig 2) — `figures/fig2_scatter.png`
[[fill]]

### 4.3 Rank disagreement, Part B (Fig 3) — `figures/fig3_ranks.png`
[[fill]]

### 4.4 Mixing curves (Fig 4) — `figures/fig4_mixing.png`
[[fill — DV lost per benchmark point gained]]

### 4.5 Patches (Fig 5) — `figures/fig5_patches.png`
[[fill]]

### 4.6 Future outcomes, Part D (Fig 6) — `figures/fig6_future.png`
[[fill]]

### 4.7 Pre-registered predictions
[[fill from results/prereg_eval.md — each supported / not supported / mixed]]

## 5. Discussion
[[fill — which benchmarks pass the null-policy test; what the gap means]]

## Appendix: Limitations and Dual-Use / Ethical Considerations

- **The LLM reader is a proxy for human readers.** DV depends on a model's estimate of
  what a careful human reader would conclude; we validate reader-to-reader agreement
  (Stage 6) and pre-fill human hand-label sheets, but human labels are future work.
- **Uniform-threshold assumption.** DV via the Brier skill score implicitly treats all
  items under one decision threshold / loss; real decisions have item-specific stakes.
- **Prompted, not natural.** The policies show the gap is *possible*, not that models drift
  into it unprompted. Part B measures how much it affects *real* rankings of unmodified
  models.
- **Compliance limits.** Abstainer and the yes-man control did not reach 80% judged
  compliance; their behaviour is bimodal / hard to score by an LLM judge that cannot see
  item difficulty. Reported as measured.
- **Dual use.** This paper shows how to game epistemic benchmarks with trivial prompts.
  We publish it because the same analysis yields the *mitigation*: the scoring patches in
  §3.5, which reward informativeness (DV) rather than the mere absence of one behaviour. We
  demonstrate exploits only on public benchmarks with public data and do not target any
  deployed system.

## What was new during the sprint

All code in this repository — the async runner, metrics, policies, dataset builders, the
DV/reader/judge pipeline, analysis and figures — was written during the sprint. No
pre-existing project code was reused.
