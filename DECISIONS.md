# DECISIONS.md

Every deviation from the spec and why. Timestamps in IST (UTC+5:30).

## 2026-11-15 ~18:10 IST — Setup

- **Timezone:** `TZ=Asia/Kolkata` does not resolve in this Git Bash (returns UTC). I compute IST as UTC+5:30 via a helper (`src/clock.py`). Current clock at setup: 2026-11-15 ~18:10 IST. Deadline Mon 16 Nov 10:00 IST = 04:30 UTC, ~15.9h from setup start.
- **`gh` CLI not installed.** Will push via plain `git` + Windows Git Credential Manager (configured at system level). If push prompts for credentials non-interactively and fails, I commit locally every stage and flag the blocker in STATUS.md.
- **Python 3.14.2.** Pre-installed: openai, pandas, numpy, scipy, matplotlib, aiohttp, tenacity, python-dotenv, huggingface_hub, requests. `datasets` not installed — will install if needed, else download dataset files directly via huggingface_hub/requests.

## Model mapping (spec name -> available Fireworks model)

Catalog is newer than the spec's names (late 2026). Full chat models only (excluded routers/spot/embedding/reranker/auto). No pricing in `/v1/models`; **assume $0.90 / 1M tokens flat** for both input and output, per spec fallback (conservative; likely overestimates spend).

**Part A subjects (2 cheapest strong chat):**
- `gpt-oss-120b` -> `accounts/fireworks/models/gpt-oss-120b` (exact match)
- `DeepSeek V4 Flash` -> `accounts/fireworks/models/deepseek-v4p1-flash` (closest; "flash" = cheap/fast tier)

**Reader + Judge (family different from both subjects):**
- `Qwen 3.7 Plus` -> `accounts/fireworks/models/qwen3p8-max` (Qwen flagship; distinct from GPT-OSS and DeepSeek families)

**Second reader / second judge (validation, Stage 6, different family again):**
- `accounts/fireworks/models/glm-5p3` (Zhipu family)

**Natural models for Part B (6 diverse families; reader/judge Qwen excluded):**
1. `accounts/fireworks/models/gpt-oss-120b` (OpenAI-oss)  — also Part A subject (baseline cache reused)
2. `accounts/fireworks/models/deepseek-v4p1-flash` (DeepSeek) — also Part A subject (cache reused)
3. `accounts/fireworks/models/minimax-m3` (MiniMax)
4. `accounts/fireworks/models/kimi-k3` (Moonshot)
5. `accounts/fireworks/models/glm-5p3` (Zhipu) — note: also 2nd reader/judge; flagged per spec
6. `accounts/fireworks/models/nemotron-3-ultra-nvfp4` (NVIDIA)

- Spec warned glm-5.2 hung on Nov 14; here we have glm-5p3 (newer). Will smoke-test for timeouts and drop if >20% time out.
- Will add a 7th natural model (minimax-m3p1-flash) only if smoke-test cost estimate leaves headroom.

## Training cutoffs

Fireworks model cards do not expose cutoff via API. Will estimate as (model `created` date - 3 months) and mark estimated, per spec, for Part D.
