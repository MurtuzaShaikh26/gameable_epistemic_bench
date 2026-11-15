"""Fill prop_answer for X=0 Omniscience items with a plausible WRONG answer (judge model).
Spot-checks 20 and records them. Idempotent: skips items already filled (cache makes re-runs free)."""
from __future__ import annotations
import asyncio
import json

import config as C
from fw_client import Runner


def messages(question, gold):
    sys = ("You generate a plausible but INCORRECT answer to a knowledge question — "
           "something that looks like it could be right in form and type, but is factually wrong.")
    user = (f"Question: {question}\nThe correct answer is: {gold}\n\n"
            "Give one plausible but WRONG answer of the same kind/format as the correct answer. "
            "Do not restate the correct answer. Reply with only the wrong answer, nothing else.")
    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]


async def main():
    path = C.SAMPLES / "omniscience.json"
    items = json.load(open(path, encoding="utf-8"))
    todo = [it for it in items if it["X"] == 0 and not it.get("prop_answer")]
    print(f"Need distractors for {len(todo)} items")
    runner = Runner(tag="distractors")
    calls = [{"model": C.JUDGE, "messages": messages(it["question"], it["gold_text"]),
              "max_tokens": 120, "meta": it["id"]} for it in todo]
    res = await runner.map(calls)
    runner.close()
    byid = {}
    for it, r in zip(todo, res):
        d = (r["content"] or "").strip().strip('"').split("\n")[0][:300]
        it["prop_answer"] = d or (it["gold_text"] + " (alt)")
        byid[it["id"]] = (it["question"], it["gold_text"], it["prop_answer"])
    json.dump(items, open(path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    # spot-check 20
    import random
    r = random.Random(C.SEED)
    sample = r.sample(list(byid), min(20, len(byid)))
    lines = ["# Omniscience distractor spot-check (20 of {})\n".format(len(byid))]
    for sid in sample:
        qq, g, d = byid[sid]
        lines.append(f"- Q: {qq[:160]}\n  - gold: {g}\n  - distractor: {d}\n")
    (C.RESULTS / "distractor_spotcheck.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote results/distractor_spotcheck.md ; spend so far $%.4f" % runner.spend.total)


if __name__ == "__main__":
    asyncio.run(main())
