"""Stage 4: Part A full runs. All policies x 2 subjects x 4 sets + reader + judge.
Cache makes it resumable for free (re-run to resume). Writes results/partA_records.json."""
from __future__ import annotations
import asyncio
import json
import sys
import time

import config as C
import experiment as EX
import scoring as SC
from fw_client import Runner
from clock import ist_str

POLICIES = ["baseline", "evader", "stonewaller", "hedger", "abstainer", "yes_man"]
SETS = ["mask", "pushback", "omniscience", "forecast"]


def load(set_name, n=None):
    items = json.load(open(C.SAMPLES / f"{set_name}.json", encoding="utf-8"))
    return items[:n] if n else items


async def main(limit=None):
    runner = Runner(tag="partA")
    items_by_set = {s: load(s, limit) for s in SETS}
    all_recs = []
    base_acc = {}
    out_path = C.RESULTS / "partA_records.json"
    for subject in C.SUBJECTS:
        for pol in POLICIES:
            for s in SETS:
                ba = base_acc.get((s, subject))
                t0 = time.time()
                recs = await EX.RUNNERS[s](runner, items_by_set[s], pol, subject, base_acc=ba)
                all_recs.extend(recs)
                if pol == "baseline":
                    base_acc[(s, subject)] = SC.round5(SC.base_accuracy(s, recs))
                # checkpoint
                json.dump(all_recs, open(out_path, "w", encoding="utf-8"),
                          default=str, ensure_ascii=False)
                runner.spend.flush()
                print(f"[{ist_str()}] A {C.short(subject)} {pol} {s}: {len(recs)} recs, "
                      f"{time.time()-t0:.0f}s, spend ${runner.spend.total:.2f}", flush=True)
                if runner.spend.total >= C.BUDGET_CAP:
                    print("!! BUDGET CAP reached, stopping new calls", flush=True)
                    runner.close()
                    return
    runner.close()
    json.dump({f"{s}:{C.short(m)}": v for (s, m), v in base_acc.items()},
              open(C.RESULTS / "base_acc.json", "w"), indent=2)
    print(f"Part A done. {len(all_recs)} records. Spend ${runner.spend.total:.2f}")


if __name__ == "__main__":
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else None
    asyncio.run(main(lim))
