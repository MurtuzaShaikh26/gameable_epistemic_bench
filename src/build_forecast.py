"""Stage 2 (Part D): build the future-outcome set from ForecastBench.

Keep binary questions resolved to 0/1 whose resolution_date is after the latest
training cutoff of the Part A subjects (deepseek-v4p1-flash ~ 2026-06-10). Sample up
to 150, balanced across sources. Stores market prob m (freeze_datetime_value) for the
optional market-proxy extension.
"""
from __future__ import annotations
import json
import random
from collections import defaultdict
from pathlib import Path

import config as C

FB = C.RAW / "forecastbench"
LATEST_CUTOFF = "2026-06-10"   # deepseek-v4p1-flash estimated cutoff (created - 3mo)
N = 150


def build():
    res = json.load(open(FB / "2026-10-25_resolution_set.json", encoding="utf-8"))
    qset = json.load(open(FB / "2026-10-25-llm.json", encoding="utf-8"))
    qbyid = {q["id"]: q for q in qset["questions"]}

    cand = []
    for r in res["resolutions"]:
        if str(r.get("resolved")) != "True":
            continue
        try:
            rt = float(r.get("resolved_to"))
        except (TypeError, ValueError):
            continue
        if rt not in (0.0, 1.0):
            continue
        rdate = r.get("resolution_date", "")
        if not rdate or rdate <= LATEST_CUTOFF:
            continue
        q = qbyid.get(r["id"])
        if not q:
            continue
        # combination questions have list ids/questions; keep simple single ones
        if isinstance(q.get("question"), list):
            continue
        m = q.get("freeze_datetime_value")
        try:
            m = float(m)
        except (TypeError, ValueError):
            m = None
        cand.append({
            "id": f"fc_{r['id'][:12]}_{rdate}",
            "fb_id": r["id"],
            "source": r.get("source", ""),
            "question": q["question"],
            "resolution_criteria": q.get("resolution_criteria", ""),
            "background": (q.get("background", "") or "")[:1500],
            "resolution_date": rdate,
            "X": int(rt),                 # actual resolution (1=YES, 0=NO)
            "market_prob": m,             # community/market prob at freeze
        })

    # balance across sources: round-robin sample
    bysrc = defaultdict(list)
    r42 = random.Random(C.SEED)
    for c in cand:
        bysrc[c["source"]].append(c)
    for s in bysrc:
        r42.shuffle(bysrc[s])
    picked = []
    srcs = sorted(bysrc)
    i = 0
    while len(picked) < N and any(bysrc.values()):
        s = srcs[i % len(srcs)]
        if bysrc[s]:
            picked.append(bysrc[s].pop())
        i += 1
        if i > 100000:
            break
    json.dump(picked, open(C.SAMPLES / "forecast.json", "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    pi = sum(c["X"] for c in picked) / len(picked) if picked else float("nan")
    from collections import Counter
    srcdist = dict(Counter(c["source"] for c in picked))
    return len(picked), pi, srcdist, len(cand)


if __name__ == "__main__":
    n, pi, srcdist, ncand = build()
    print(f"ForecastBench: {n} items (from {ncand} eligible), pi={pi:.3f}")
    print("sources:", srcdist)
