"""Stage 5: Part B natural-model runs. Baseline policy only x natural models x 4 sets.
Cached subject baseline calls for gpt-oss-120b / deepseek (shared with Part A) are free.
Writes results/partB_records.json."""
from __future__ import annotations
import asyncio
import json
import sys
import time

import config as C
import experiment as EX
from fw_client import Runner
from clock import ist_str

SETS = ["mask", "pushback", "omniscience", "forecast"]


def load(set_name, n=None):
    items = json.load(open(C.SAMPLES / f"{set_name}.json", encoding="utf-8"))
    return items[:n] if n else items


async def main(limit=None, models=None):
    models = models or C.NATURAL
    runner = Runner(tag="partB")
    items_by_set = {s: load(s, limit) for s in SETS}
    all_recs = []
    out_path = C.RESULTS / "partB_records.json"
    for model in models:
        for s in SETS:
            t0 = time.time()
            recs = await EX.RUNNERS[s](runner, items_by_set[s], "baseline", model, base_acc=None)
            for r in recs:
                r["natural_model"] = C.short(model)
            all_recs.extend(recs)
            json.dump(all_recs, open(out_path, "w", encoding="utf-8"),
                      default=str, ensure_ascii=False)
            runner.spend.flush()
            print(f"[{ist_str()}] B {C.short(model)} {s}: {len(recs)} recs, "
                  f"{time.time()-t0:.0f}s, spend ${runner.spend.total:.2f}", flush=True)
            if runner.spend.total >= C.BUDGET_CAP:
                print("!! BUDGET CAP reached, stopping", flush=True)
                runner.close()
                return
    runner.close()
    print(f"Part B done. {len(all_recs)} records. Spend ${runner.spend.total:.2f}")


if __name__ == "__main__":
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else None
    asyncio.run(main(lim))
