"""Stage 6: reader & judge validation.
- Reader: 50 items across sets/policies, q from a 2nd reader (diff family); MAD + Spearman.
- Judge: 50 MASK commit verdicts + 50 Omniscience grades from a 2nd judge; agreement.
- Pre-fill human hand-label CSVs with an empty human column (non-blocking)."""
from __future__ import annotations
import asyncio
import csv
import json
import random

import numpy as np

import config as C
import pipeline as PL
import metrics as mt
from fw_client import Runner


def load_recs():
    A = json.load(open(C.RESULTS / "partA_records.json", encoding="utf-8"))
    return A


def pick_reader_items(A, n=50):
    """Collect (proposition, answer, q1) triples from Part A records."""
    pool = []
    samp = json.load  # noqa
    masks = {it["id"]: it for it in json.load(open(C.SAMPLES / "mask.json", encoding="utf-8"))}
    push = {it["id"]: it for it in json.load(open(C.SAMPLES / "pushback.json", encoding="utf-8"))}
    omni = {it["id"]: it for it in json.load(open(C.SAMPLES / "omniscience.json", encoding="utf-8"))}
    for r in A:
        if r["set"] == "mask" and r.get("q") is not None and r["id"] in masks:
            pool.append(("mask", r["id"], r["policy"], PL.mask_proposition(masks[r["id"]]),
                         r.get("pressure_answer", ""), r["q"]))
        elif r["set"] == "omniscience" and r.get("q") is not None and r["id"] in omni:
            pool.append(("omniscience", r["id"], r["policy"], PL.omniscience_proposition(omni[r["id"]]),
                         r.get("primary_answer", ""), r["q"]))
        elif r["set"] == "pushback" and r.get("q_evidence") is not None and r["id"] in push:
            pool.append(("pushback", r["id"], r["policy"], PL.pushback_proposition(push[r["id"]]),
                         "", r["q_evidence"]))
    rng = random.Random(C.SEED)
    rng.shuffle(pool)
    return pool[:n]


async def main():
    A = load_recs()
    runner = Runner(tag="validate")
    # ---- reader validation ----
    items = pick_reader_items(A, 50)
    calls = [{"model": C.READER2, "messages": PL.reader_messages(p, a),
              "max_tokens": C.READER_MAX_TOKENS, "meta": iid} for (_, iid, _, p, a, _) in items]
    res = await runner.map(calls)
    rows = []
    q1s, q2s = [], []
    for (s, iid, pol, prop, ans, q1), r in zip(items, res):
        q2 = PL.parse_q(r["content"])
        rows.append({"set": s, "id": iid, "policy": pol, "proposition": prop[:200],
                     "answer": (ans or "")[:200], "reader1_q": q1, "reader2_q": q2, "human_q": ""})
        if q2 is not None:
            q1s.append(q1); q2s.append(q2)
    with open(C.RESULTS / "handlabel_reader.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    mad = float(np.mean(np.abs(np.array(q1s) - np.array(q2s)))) if q1s else float("nan")
    rho = mt.spearman(q1s, q2s) if len(q1s) > 2 else float("nan")

    # ---- judge validation ----
    masks = {it["id"]: it for it in json.load(open(C.SAMPLES / "mask.json", encoding="utf-8"))}
    mask_recs = [r for r in A if r["set"] == "mask" and r.get("pressure_answer")][:50]
    jcalls = [{"model": C.JUDGE2, "messages": PL.mask_commit_judge_messages(
        masks[r["id"]]["proposition_raw"], masks[r["id"]]["pivotal_opts"], r["pressure_answer"]),
        "max_tokens": C.JUDGE_MAX_TOKENS, "meta": r["id"]} for r in mask_recs if r["id"] in masks]
    jres = await runner.map(jcalls)
    jrows, agree = [], 0
    for r, jr in zip(mask_recs, jres):
        v2 = PL.parse_mask_commit(jr["content"])
        v1 = r["pressure_commit"]
        agree += (v1 == v2)
        jrows.append({"type": "mask_commit", "id": r["id"], "judge1": v1, "judge2": v2, "human": ""})
    mask_agree = agree / len(jrows) if jrows else float("nan")

    omni = {it["id"]: it for it in json.load(open(C.SAMPLES / "omniscience.json", encoding="utf-8"))}
    omni_recs = [r for r in A if r["set"] == "omniscience" and r.get("primary_answer")][:50]
    ocalls = [{"model": C.JUDGE2, "messages": PL.omniscience_grade_messages(
        omni[r["id"]]["question"], omni[r["id"]]["gold_text"], r["primary_answer"]),
        "max_tokens": C.JUDGE_MAX_TOKENS, "meta": r["id"]} for r in omni_recs if r["id"] in omni]
    ores = await runner.map(ocalls)
    agree2 = 0
    for r, orr in zip(omni_recs, ores):
        v2 = PL.parse_grade(orr["content"]); v1 = r["grade"]
        agree2 += (v1 == v2)
        jrows.append({"type": "omni_grade", "id": r["id"], "judge1": v1, "judge2": v2, "human": ""})
    omni_agree = agree2 / len(omni_recs) if omni_recs else float("nan")
    with open(C.RESULTS / "handlabel_judge.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["type", "id", "judge1", "judge2", "human"])
        w.writeheader(); w.writerows(jrows)
    runner.close()

    report = (f"# Validation (Stage 6)\n\n"
              f"Reader1={C.short(C.READER)} vs Reader2={C.short(C.READER2)} on {len(q1s)} items:\n"
              f"- mean |q1-q2| = {mad:.3f}\n- Spearman ρ = {rho:.3f}  (gate: >=0.6)\n\n"
              f"Judge1={C.short(C.JUDGE)} vs Judge2={C.short(C.JUDGE2)}:\n"
              f"- MASK commit agreement = {mask_agree:.1%} (n={len(mask_recs)})\n"
              f"- Omniscience grade agreement = {omni_agree:.1%} (n={len(omni_recs)})\n\n"
              f"Hand-label CSVs written with empty human columns (non-blocking): "
              f"results/handlabel_reader.csv, results/handlabel_judge.csv\n")
    (C.RESULTS / "validation.md").write_text(report, encoding="utf-8")
    print(report)
    if not np.isnan(rho) and rho < 0.6:
        print("!! Reader Spearman < 0.6 — flag prominently in draft.md as a limitation")


if __name__ == "__main__":
    asyncio.run(main())
