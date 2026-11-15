"""Stage 2: sample all item sets with seed 42, write data/samples/*.json and DATA.md.

Known-answer sets: MASK, Pushback (sycophancy are_you_sure), AA-Omniscience.
Part D (ForecastBench) handled by build_forecast.py (cuttable).

Balance pi ~ 0.5: for each set, half the items get the proposition stated with the
gold answer (X=1) and half with a plausible wrong answer (X=0), chosen by seed 42.
"""
from __future__ import annotations
import ast
import json
import re
import random
from pathlib import Path

import pandas as pd

import config as C

RAW = C.RAW
SAMP = C.SAMPLES
rng = random.Random(C.SEED)

PIVOTAL_RE = re.compile(r"<pivotal>(.*?)</pivotal>", re.DOTALL)


def assign_balance(ids):
    """Deterministically split ids into X=1 (gold) and X=0 (wrong) halves, seed 42."""
    r = random.Random(C.SEED)
    order = list(ids)
    r.shuffle(order)
    half = len(order) // 2
    xval = {}
    for i, _id in enumerate(order):
        xval[_id] = 1 if i < half else 0
    return xval


# ---------------- MASK ----------------
def build_mask(n=200):
    splits = ["known_facts", "provided_facts", "continuations", "disinformation",
              "doubling_down_known_facts", "statistics"]
    rows = []
    for sp in splits:
        p = RAW / "mask" / sp / "test-00000-of-00001.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        for _, r in df.iterrows():
            prop = str(r.get("proposition", ""))
            fgt = str(r.get("formatted_ground_truth", ""))
            m = PIVOTAL_RE.search(prop)
            # need a binary pivotal proposition with 2 options and a known A/B truth
            if not m or fgt not in ("A", "B"):
                continue
            opts = [o.strip() for o in m.group(1).split("/")]
            if len(opts) != 2:
                continue
            rows.append({
                "id": f"mask_{sp}_{r['task_id']}",
                "split": sp,
                "task_id": r["task_id"],
                "system_prompt": r.get("system_prompt") or "",
                "user_prompt": r["user_prompt"],
                "proposition_raw": prop,
                "pivotal_opts": opts,          # [A_option, B_option]
                "truth_letter": fgt,            # 'A' or 'B'
                "ground_truth": r.get("ground_truth", ""),
                "belief_elicit_1": r.get("belief_elicit_1", ""),
            })
    rng2 = random.Random(C.SEED)
    rng2.shuffle(rows)
    rows = rows[:n]
    xval = assign_balance([r["id"] for r in rows])
    for r in rows:
        r["X"] = xval[r["id"]]
        true_opt = r["pivotal_opts"][0] if r["truth_letter"] == "A" else r["pivotal_opts"][1]
        wrong_opt = r["pivotal_opts"][1] if r["truth_letter"] == "A" else r["pivotal_opts"][0]
        chosen = true_opt if r["X"] == 1 else wrong_opt
        # declarative proposition with pivotal resolved
        r["proposition"] = PIVOTAL_RE.sub(chosen, r["proposition_raw"]).strip()
    json.dump(rows, open(SAMP / "mask.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    pi = sum(r["X"] for r in rows) / len(rows)
    return len(rows), pi


# ---------------- Pushback (sycophancy are_you_sure) ----------------
def _parse_base(s):
    if isinstance(s, dict):
        return s
    try:
        return ast.literal_eval(s)
    except Exception:
        try:
            return json.loads(s)
        except Exception:
            return None


def _opts_from_answers(ans_str):
    """Parse '\n(A)5\n(B)6\n(C)7...' into {'A':'5','B':'6',...}."""
    out = {}
    for m in re.finditer(r"\(([A-E])\)\s*(.*)", ans_str):
        out[m.group(1)] = m.group(2).strip()
    return out


def build_pushback(n=200):
    recs = []
    with open(RAW / "sycophancy" / "are_you_sure.jsonl", encoding="utf-8") as f:
        for line in f:
            recs.append(json.loads(line))
    cand = []
    # MC datasets with unambiguous gold letter
    good_sets = {"aqua_mc", "math_mc_cot", "mmlu_mc_cot", "truthful_qa_mc"}
    for i, x in enumerate(recs):
        b = _parse_base(x.get("base"))
        if not b:
            continue
        if b.get("dataset") not in good_sets:
            continue
        letter = b.get("correct_letter")
        opts = _opts_from_answers(b.get("answers", ""))
        if not letter or letter not in opts or len(opts) < 2:
            continue
        gold_text = opts[letter]
        wrong_letters = [k for k in opts if k != letter]
        if not wrong_letters:
            continue
        cand.append({
            "id": f"push_{b['dataset']}_{i}",
            "dataset": b["dataset"],
            "question": b["question"],
            "options": opts,
            "gold_letter": letter,
            "gold_text": gold_text,
            "wrong_letters": wrong_letters,
        })
    rng3 = random.Random(C.SEED)
    rng3.shuffle(cand)
    cand = cand[:n]
    xval = assign_balance([c["id"] for c in cand])
    r4 = random.Random(C.SEED + 1)
    for c in cand:
        c["X"] = xval[c["id"]]
        # proposition stated as "The correct answer to <q> is <answer>."
        if c["X"] == 1:
            c["prop_answer"] = c["gold_text"]
        else:
            wl = r4.choice(c["wrong_letters"])
            c["prop_answer"] = c["options"][wl]
    json.dump(cand, open(SAMP / "pushback.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    pi = sum(c["X"] for c in cand) / len(cand)
    return len(cand), pi


# ---------------- AA-Omniscience ----------------
def build_omniscience(n=250):
    df = pd.read_csv(RAW / "omniscience" / "AA-Omniscience_dataset_public.csv")
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "id": f"omni_{r['question_id']}",
            "question_id": int(r["question_id"]),
            "domain": r.get("domain", ""),
            "topic": r.get("topic", ""),
            "question": r["question"],
            "gold_text": str(r["answer"]),
        })
    rng5 = random.Random(C.SEED)
    rng5.shuffle(rows)
    rows = rows[:n]
    xval = assign_balance([r["id"] for r in rows])
    for r in rows:
        r["X"] = xval[r["id"]]
        # prop_answer for X=1 is gold; for X=0 a distractor filled later by generate_distractors.py
        r["prop_answer"] = r["gold_text"] if r["X"] == 1 else None
    json.dump(rows, open(SAMP / "omniscience.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    pi = sum(r["X"] for r in rows) / len(rows)
    n_need_distractor = sum(1 for r in rows if r["X"] == 0)
    return len(rows), pi, n_need_distractor


if __name__ == "__main__":
    print("Building datasets (seed 42)...")
    nm, pim = build_mask()
    print(f"MASK: {nm} items, pi={pim:.3f}")
    npush, pipush = build_pushback()
    print(f"Pushback: {npush} items, pi={pipush:.3f}")
    no, pio, nd = build_omniscience()
    print(f"Omniscience: {no} items, pi={pio:.3f} ({nd} need generated distractors for X=0)")
    print("Done. Wrote data/samples/{mask,pushback,omniscience}.json")
