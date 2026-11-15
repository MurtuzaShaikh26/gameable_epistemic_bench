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
Across 2 subjects × 6 policies × 4 item sets the exploits land hard: a one-line **evader**
reaches *perfect* MASK honesty (1.00 vs 0.81 baseline) and flips the AA-Omniscience Index
from −24.4 to +3.0 (best of any policy), while its Decision Value stays at ≈0 — it is
useless. The **hedger** drives ECE to 0.037 (vs 0.43 baseline) with a single constant
confidence. Ranking 6 unmodified models, benchmark rank and DV rank disagree sharply on the
non-sycophancy benchmark (Spearman ρ = −0.37) and only loosely elsewhere (ρ = 0.83 MASK,
0.94 Omniscience, 0.20 forecast). One natural model (`glm-5p3`) behaves like a prompted
evader unprompted — top of every benchmark, bottom on informativeness. We propose one small
scoring patch per benchmark; the Pushback patch flips its DV rank-correlation from −0.81 to
+0.53 and the patched MASK score strips the evader of its gamed 1.00. On forecasting
questions that resolved after each model's training cutoff, DV collapses to ≈0 or below for
every model (one refuses to forecast at all), confirming the benchmarks reward recall rather
than reasoning. The gap is a property of the *scoring rule*, not of any model: the patches
are the mitigation.

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
- **B (real models).** We rank 6 unmodified chat models on each benchmark and on DV,
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

Each cell is *published score | DV [95% bootstrap CI]*, averaged over both subjects
(Part A). The pattern is consistent across three of the four benchmarks: **the degenerate
policies match or beat the baseline on the published metric while their DV sits at or below
zero.**

| policy | MASK honesty | Pushback (evidence) | Omniscience Index | Forecast |
|---|---|---|---|---|
| baseline | 0.81 \| DV −0.40 | 0.89 \| DV +0.77 | −24.4 \| DV −0.27 | −0.24 \| DV −0.09 |
| evader | **1.00** \| DV +0.03 | **1.00** \| DV −0.01 | **+3.0** \| DV −0.04 | — \| DV −0.01 |
| stonewaller | 0.85 \| DV −0.58 | **1.00** \| DV +0.32 | −27.8 \| DV −0.31 | −0.23 \| DV −0.11 |
| hedger | 0.80 \| DV −0.17 | 0.91 \| DV +0.67 | −23.6 \| DV −0.31 | −0.10 \| DV −0.07 |
| abstainer | 0.98 \| DV +0.04 | 0.99 \| DV +0.08 | **−2.2** \| DV −0.34 | — \| DV −0.02 |
| yes_man | 0.81 \| DV −0.56 | 0.55 \| DV +0.89 | −28.2 \| DV −0.34 | −0.12 \| DV −0.01 |

The headline cells: the **evader** scores a perfect 1.00 on both MASK honesty and Pushback
non-sycophancy, and lifts the Omniscience Index from −24.4 to +3.0 (the single best value in
the table) — yet its DV never exceeds +0.03, meaning a careful reader learns essentially
nothing from its answers. The **abstainer** likewise improves Omniscience (−2.2 vs −24.4) at
DV ≈ 0. The **hedger** nearly matches baseline on every published metric with a fixed
confidence string. Only the Pushback *baseline* keeps a high DV (+0.77), because answering
correctly and holding under pressure is exactly what that benchmark's score and DV both
want for a *competent* model — which is why the useless policies that also score 1.00 are so
revealing. The yes-man control behaves as designed: it is the one policy that *tanks* the
Pushback score (0.55), because it caves to every "are you sure?".

### 4.2 Benchmark vs DV (Fig 2) — `figures/fig2_scatter.png`

Plotting every (policy × subject × benchmark) cell as published-score vs DV shows no
positive relationship for MASK or Omniscience: the points with the best published scores
are the DV ≈ 0 degenerate policies, so the scatter is flat-to-negative. Pushback is the
exception — there, higher score does track higher DV among the *answering* conditions,
because the benchmark rewards a behaviour (holding a correct answer) that is genuinely
informative; the degenerate policies reach the same score by a different, uninformative
route (never committing), which DV exposes.

### 4.3 Rank disagreement, Part B (Fig 3) — `figures/fig3_ranks.png`

Ranking the 6 unmodified models on each benchmark against their DV rank (Spearman ρ):

| benchmark | ρ(benchmark, DV) | interpretation |
|---|---|---|
| MASK | +0.83 | loose agreement |
| Pushback | **−0.37** | **disagree — near-opposite order** |
| Omniscience | +0.94 | agreement |
| Forecast | +0.20 | essentially unranked |

The Pushback benchmark ranks models in nearly the reverse of DV order. The cause is a single
natural model, **`glm-5p3`**, which behaves like a prompted evader without being asked to:
it posts the highest MASK honesty (0.995), the highest Pushback resistance (0.967), and the
best Omniscience Index (+11.6) and ECE (0.095) of all six models — while committing to an
answer on MASK only 8.5% of the time, accepting a *true* correction only 8% of the time, and
answering Omniscience questions with only 16% accuracy. It wins the leaderboards by declining
to be useful. DV ranks it near the bottom on informativeness, which is why the two orders
diverge.

### 4.4 Mixing curves (Fig 4) — `figures/fig4_mixing.png`

Interpolating each subject from baseline toward a degenerate policy and tracking (benchmark
score, DV) traces how much DV is spent per point of benchmark gained. On Pushback the trade
is steep and adverse — moving toward the evader or abstainer *raises* the benchmark score
while DV falls by roughly 9–11 DV-points per score-point (`results/mixing_slopes.json`). On
Omniscience the degenerate policies move the Index substantially with almost no DV change
(slopes ≈ 0), i.e. pure score inflation. The hedger is the benign case: it mostly shifts ECE
without moving DV much, because it changes only the confidence string, not the answer.

### 4.5 Patches (Fig 5) — `figures/fig5_patches.png`

Rank-correlation with DV across the 6 natural models, original score vs patched score
(`results/patch_corr.json`):

- **Pushback — the patch works cleanly.** Original resistance correlates −0.81 with DV
  (resisting pushback, alone, anti-predicts informativeness because the stonewaller/evader
  resist *everything*). The patch — harmonic mean of resistance and correction-acceptance —
  flips this to **+0.53**, and in Part A it strips the stonewaller of credit for refusing
  true corrections (its evidence-condition DV is +0.32 vs the baseline's +0.77).
- **MASK — the patch is validated by Part A, not by the Part B ranking.** The patched score
  (honesty × commitment) correctly denies the evader its gamed 1.00 (the evader's commitment
  ≈ 0). Across the 6 natural models its rank-correlation with DV is −0.89 vs +0.85 for raw
  honesty; with n = 6 this is noisy, and among *natural* models committing more happens to
  co-occur with both more informativeness and more lying, so the two diverge. The patch's job
  is to close the prompted exploit, which it does.
- **Omniscience — Brier/DV reporting and the Kalai penalty both help.** The raw Index
  already correlates +0.87 with DV among natural models; the Kalai-style t/(1−t) penalty at
  t = 0.75 correlates +0.77 and, unlike the Index, assigns the fixed-confidence hedger no
  skill. Reporting Brier skill alongside the Index removes the free lunch the abstainer/evader
  get from never being wrong.

### 4.6 Future outcomes, Part D (Fig 6) — `figures/fig6_future.png`

On 150 ForecastBench questions that resolved *after* every subject's training cutoff, DV
collapses for all six models: DV ranges from −0.01 to −0.26, i.e. no model beats the item
base rate, and `glm-5p3` abstains on 100% of questions (refusing to forecast at all).
Self-reported-confidence DV (`self_dv`) is worse still (e.g. −0.82 for `nemotron-3-ultra`),
showing the models are over-confident about futures they cannot know. This is the control
the whole paper needs: where memorisation cannot help, the informativeness the benchmarks
are supposed to be proxies for simply is not there — so a benchmark that still hands out a
high score is measuring recall, not epistemics.

### 4.7 Pre-registered predictions

From `PREREG.md`, evaluated in `results/prereg_eval.md`:

1. Evader MASK honesty ≥ baseline — **SUPPORTED** (1.00 vs 0.81).
2. Stonewaller has the lowest Pushback abandonment — **NOT SUPPORTED**: the evader tied it
   at 0.0 (it never gives an answer to abandon), an honest miss that itself illustrates the
   point — "never abandons" is trivially won by "never answers".
3. Hedger has the lowest ECE — **SUPPORTED** (0.037 vs 0.32–0.47 for the rest).
4. Abstainer Omniscience Index > baseline when baseline < 0 — **SUPPORTED** (−2.2 vs −24.4).
5. (Descriptive) Baseline DV exceeds evader/abstainer DV on every known-answer set —
   confirmed on MASK, Pushback, Omniscience.
6. Stonewaller evidence-condition DV < baseline — **SUPPORTED** (+0.32 vs +0.77): refusing
   *true* corrections costs real informativeness the benchmark score does not see.
7. Benchmark-vs-DV rank agreement is weakest on Pushback — **SUPPORTED** (ρ = −0.37, the
   only negative).

4 of 5 directional predictions supported; the one miss strengthens the thesis rather than
weakening it.

### 4.8 Prompt-rotation robustness (`results/rotation.md`)

To rule out that the exploit is an artifact of one specific prompt wording, we reran two
alternate phrasings each of the evader and abstainer — same behaviour, different words —
through the identical pipeline on MASK and Omniscience (subject `gpt-oss-120b`, full sets).
**All four alternates reproduced both exploits.** Every evader/abstainer rewording held
MASK honesty at ≥ 0.96 with DV ≤ +0.12, and every rewording lifted the Omniscience Index
far above the −52.8 single-subject baseline (to between −16 and +1) at DV ≤ 0. The exploit
tracks the *behaviour* (never commit / always abstain), not the phrasing, which is what the
null-policy framing predicts.

## 5. Discussion

**Which benchmarks pass the null-policy test?** None cleanly. MASK and AA-Omniscience fail
outright: a one-line evader reaches the top score while contributing no information, because
honesty-as-"never-lies" and Index-as-"never-wrong" are both maximised by never committing.
The sycophancy "are you sure?" benchmark fails in a subtler and more dangerous way — its
score *anti-correlates* with DV across real models (ρ = −0.37), because "resists pushback"
and "accepts correction" are scored as the same virtue when they are opposites; a model that
resists everything wins. Calibration-via-ECE fails trivially to the hedger. The forecasting
setting is the cleanest: with memorisation removed, every model's DV sits at ≈0, which is the
honest reading of what these models can do on genuinely unknown questions.

**What the gap means.** The failures are properties of the *scoring rules*, not of the
models — the same model scores very differently under the published metric and under DV, and
a natural model (`glm-5p3`) already exploits the gap without any prompting. The fix is not a
new benchmark but a scoring change, exactly as Kalai et al. (2025) argue: reward
informativeness (DV, or the per-benchmark patches in §3.5) rather than the mere absence of
one bad behaviour. The patches are cheap, local, and — for Pushback demonstrably, for MASK
and Omniscience by construction — close the specific hole each benchmark leaves open.

## Appendix: Limitations and Dual-Use / Ethical Considerations

- **The LLM reader is a proxy for human readers — validated, with a known noise floor.** DV
  depends on a model's estimate of what a careful human reader would conclude, so we validated
  it two ways. (i) *Model-vs-model*: reader `qwen3p8-max` vs `kimi-k3` on 50 held-out items
  gives Spearman ρ = 0.511 (below our 0.6 gate). (ii) *Human-anchored*: the project owner
  hand-labelled the same 50 reader items, 47 MASK commit/evade judgments, and 50 Omniscience
  grades (`results/validation_human.md`). Against the human, the DV reader does **better** than
  the second model — mean |Δq| = **0.134**, Pearson r = 0.687, **Spearman ρ = 0.641 (above the
  0.6 gate)**, with 76% three-way directional agreement (false-/uninformative-/true-leaning).
  Human grading of AA-Omniscience answers confirms the judge almost perfectly (98% agreement,
  Cohen κ = 0.97). The one genuinely noisy component is the **MASK commit-vs-evade** judgment
  (human-vs-model 81%, κ = 0.63): disagreements concentrate on refusal-with-explanation
  answers, where the model scores a stated fact as a commitment while a human reads the
  refusal framing as evasion. We therefore still read DV *magnitudes* as directional and rest
  conclusions on **rank** and on **large** gaps, and we flag the MASK commitment rate (and thus
  the MASK patch) as the result most sensitive to judge noise. Filled label sheets:
  `results/handlabel_filled_reader.csv`, `handlabel_filled_judge.csv`.
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
