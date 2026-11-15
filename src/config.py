"""Central configuration: models, paths, constants."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
DATA = ROOT / "data"
SAMPLES = DATA / "samples"
RAW = DATA / "raw"
FIGURES = ROOT / "figures"
CACHE = ROOT / ".cache"
for p in (RESULTS, DATA, SAMPLES, RAW, FIGURES, CACHE):
    p.mkdir(parents=True, exist_ok=True)

FIREWORKS_API_KEY = os.environ.get("FIREWORKS_API_KEY", "")
BASE_URL = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")

SEED = 42

# --- Models (see DECISIONS.md for mapping rationale) ---
M = "accounts/fireworks/models/"

SUBJECTS = [            # Part A subjects
    M + "gpt-oss-120b",
    M + "deepseek-v4p1-flash",
]
READER = M + "qwen3p8-max"       # DV reader
JUDGE = M + "qwen3p8-max"        # policy-compliance / MASK / Omniscience judge
READER2 = M + "kimi-k3"          # second reader (validation; different family, supports no-think)
JUDGE2 = M + "minimax-m3"        # second judge (validation; different family, supports no-think)

# Per-model reasoning control. Value passed as reasoning_effort when not None.
# 'none' disables thinking (cheap, reliable short outputs); 'low' for gpt-oss (rejects 'none');
# None = do not pass the param (thinking-only models like glm-5p3).
MODEL_REASONING = {
    M + "gpt-oss-120b": "low",
    M + "deepseek-v4p1-flash": "none",
    M + "qwen3p8-max": "none",
    M + "minimax-m3": "none",
    M + "kimi-k3": "none",
    M + "nemotron-3-ultra-nvfp4": "none",
    M + "glm-5p3": None,                 # thinking-only; cannot disable
    M + "minimax-m3p1-flash": "none",
}

NATURAL = [            # Part B natural models (baseline only)
    M + "gpt-oss-120b",
    M + "deepseek-v4p1-flash",
    M + "minimax-m3",
    M + "kimi-k3",
    M + "glm-5p3",
    M + "nemotron-3-ultra-nvfp4",
]

# Optional extra natural model if budget allows
NATURAL_OPTIONAL = [M + "minimax-m3p1-flash"]

# --- Call parameters ---
SUBJECT_MAX_TOKENS = 600
READER_MAX_TOKENS = 16
JUDGE_MAX_TOKENS = 200
TEMPERATURE = 0.0
TEMPERATURE_FALLBACK = 0.3

# --- Budget / spend ---
PRICE_PER_MTOK = 0.90   # flat $/1M tokens (in and out), spec fallback
BUDGET_CAP = 90.0
BUDGET_TARGET = 45.0

CONCURRENCY = 8

# Short labels for reporting
def short(model_id: str) -> str:
    return model_id.split("/")[-1]
