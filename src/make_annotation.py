"""Regenerate self-contained hand-label sheets a human can actually fill.
No API calls: reads results/partA_records.json + data/samples/*. Full (untruncated) text,
with the model guesses and an empty human column last. Deterministic (seed 42)."""
from __future__ import annotations
import csv
import json
import random

import config as C
import pipeline as PL

SEED = 42
A = json.load(open(C.RESULTS / "partA_records.json", encoding="utf-8"))
mask = {it["id"]: it for it in json.load(open(C.SAMPLES / "mask.json", encoding="utf-8"))}
omni = {it["id"]: it for it in json.load(open(C.SAMPLES / "omniscience.json", encoding="utf-8"))}


def clean(s):
    return " ".join((s or "").split())


# ---------- READER sheet: your probability the proposition is TRUE, given only the answer ----------
reader_pool = []
for r in A:
    if r["set"] == "mask" and r.get("q") is not None and r.get("pressure_answer") and r["id"] in mask:
        reader_pool.append({
            "set": "mask", "id": r["id"], "policy": r["policy"],
            "proposition": clean(PL.mask_proposition(mask[r["id"]])),
            "answer_the_AI_gave": clean(r["pressure_answer"]),
            "model_reader_q": r["q"],
        })
    elif r["set"] == "omniscience" and r.get("q") is not None and r.get("primary_answer") and r["id"] in omni:
        reader_pool.append({
            "set": "omniscience", "id": r["id"], "policy": r["policy"],
            "proposition": clean(PL.omniscience_proposition(omni[r["id"]])),
            "answer_the_AI_gave": clean(r["primary_answer"]),
            "model_reader_q": r["q"],
        })
rng = random.Random(SEED)
rng.shuffle(reader_pool)
# balance: ~half mask, ~half omniscience, 50 total
mk = [x for x in reader_pool if x["set"] == "mask"][:25]
om = [x for x in reader_pool if x["set"] == "omniscience"][:25]
reader_rows = mk + om
rng.shuffle(reader_rows)
for x in reader_rows:
    x["YOUR_probability_0_to_1"] = ""   # <-- you fill this

with open(C.RESULTS / "annotate_reader.csv", "w", newline="", encoding="utf-8") as f:
    cols = ["set", "id", "policy", "proposition", "answer_the_AI_gave",
            "model_reader_q", "YOUR_probability_0_to_1"]
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    w.writerows(reader_rows)

# ---------- JUDGE sheet A: did the AI COMMIT or EVADE on MASK? ----------
judge_rows = []
mask_recs = [r for r in A if r["set"] == "mask" and r.get("pressure_answer") and r["id"] in mask]
rng.shuffle(mask_recs)
for r in mask_recs[:50]:
    it = mask[r["id"]]
    judge_rows.append({
        "task": "commit_or_evade", "id": r["id"], "policy": r["policy"],
        "question_or_claim": clean(it["proposition_raw"]),
        "options": clean(str(it.get("pivotal_opts", ""))),
        "answer_the_AI_gave": clean(r["pressure_answer"]),
        "model_judge": r["pressure_commit"],
        "YOUR_label_COMMIT_or_EVADE": "",   # <-- you fill this
    })
with open(C.RESULTS / "annotate_judge_mask.csv", "w", newline="", encoding="utf-8") as f:
    cols = ["task", "id", "policy", "question_or_claim", "options",
            "answer_the_AI_gave", "model_judge", "YOUR_label_COMMIT_or_EVADE"]
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    w.writerows(judge_rows)

# ---------- JUDGE sheet B: was the AI's Omniscience answer correct / incorrect / abstain? ----------
ojudge_rows = []
omni_recs = [r for r in A if r["set"] == "omniscience" and r.get("primary_answer") and r["id"] in omni]
rng.shuffle(omni_recs)
for r in omni_recs[:50]:
    it = omni[r["id"]]
    ojudge_rows.append({
        "task": "grade", "id": r["id"], "policy": r["policy"],
        "question": clean(it["question"]),
        "correct_answer": clean(it["gold_text"]),
        "answer_the_AI_gave": clean(r["primary_answer"]),
        "model_judge": r["grade"],
        "YOUR_label_correct_incorrect_abstain": "",   # <-- you fill this
    })
with open(C.RESULTS / "annotate_judge_omniscience.csv", "w", newline="", encoding="utf-8") as f:
    cols = ["task", "id", "policy", "question", "correct_answer",
            "answer_the_AI_gave", "model_judge", "YOUR_label_correct_incorrect_abstain"]
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    w.writerows(ojudge_rows)

print(f"annotate_reader.csv: {len(reader_rows)} rows (mask {len(mk)} + omniscience {len(om)})")
print(f"annotate_judge_mask.csv: {len(judge_rows)} rows")
print(f"annotate_judge_omniscience.csv: {len(ojudge_rows)} rows")
