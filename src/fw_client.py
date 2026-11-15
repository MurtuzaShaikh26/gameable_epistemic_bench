"""Async Fireworks runner: disk cache, backoff, JSONL log, spend tracking.

A call is NEVER paid for twice: results are cached on disk keyed by the hash of
(model, messages, max_tokens, temperature). Cache hits cost nothing.
"""
from __future__ import annotations
import asyncio
import hashlib
import json
import time
from pathlib import Path

from openai import AsyncOpenAI

import config as C

_lock = asyncio.Lock()


def _key(model, messages, max_tokens, temperature):
    payload = json.dumps(
        {"model": model, "messages": messages, "max_tokens": max_tokens,
         "temperature": temperature},
        sort_keys=True, ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class Spend:
    """Tracks tokens in/out per model and estimated $ spend."""
    def __init__(self, path: Path):
        self.path = path
        self.data = {"by_model": {}, "price_per_mtok": C.PRICE_PER_MTOK,
                     "n_calls": 0, "n_cached": 0, "total_usd": 0.0}
        if path.exists():
            try:
                self.data = json.load(open(path))
            except Exception:
                pass
        self.data.setdefault("by_model", {})

    def add(self, model, pin, pout, cached=False):
        m = self.data["by_model"].setdefault(
            model, {"tokens_in": 0, "tokens_out": 0, "calls": 0, "cached": 0, "usd": 0.0})
        if cached:
            m["cached"] += 1
            self.data["n_cached"] += 1
            return
        m["tokens_in"] += pin
        m["tokens_out"] += pout
        m["calls"] += 1
        usd = (pin + pout) / 1e6 * C.PRICE_PER_MTOK
        m["usd"] += usd
        self.data["n_calls"] += 1
        self.data["total_usd"] = sum(v["usd"] for v in self.data["by_model"].values())

    def flush(self):
        json.dump(self.data, open(self.path, "w"), indent=2)

    @property
    def total(self):
        return self.data["total_usd"]


class Runner:
    def __init__(self, concurrency=None, tag="run"):
        self.sem = asyncio.Semaphore(concurrency or C.CONCURRENCY)
        self.client = AsyncOpenAI(api_key=C.FIREWORKS_API_KEY, base_url=C.BASE_URL,
                                  timeout=120.0, max_retries=0)
        self.cache = C.CACHE
        self.spend = Spend(C.RESULTS / "spend.json")
        self.log_path = C.RESULTS / f"calls_{tag}.jsonl"
        self._logf = open(self.log_path, "a", encoding="utf-8")
        self._since_flush = 0

    def _cache_path(self, key):
        return self.cache / f"{key}.json"

    async def chat(self, model, messages, max_tokens, temperature=None, meta=None):
        """messages: full OpenAI-style list (incl. any system message).
        Returns dict: {content, usage, cached, error, model, meta}."""
        if temperature is None:
            temperature = C.TEMPERATURE
        key = _key(model, messages, max_tokens, temperature)
        cp = self._cache_path(key)
        if cp.exists():
            try:
                cached = json.load(open(cp, encoding="utf-8"))
                async with _lock:
                    self.spend.add(model, 0, 0, cached=True)
                cached["cached"] = True
                cached["meta"] = meta
                return cached
            except Exception:
                pass

        async with self.sem:
            result = await self._call_with_retry(model, messages, max_tokens, temperature)
        result["meta"] = meta
        result["model"] = model
        # write cache only on success (has content and no error)
        if result.get("error") is None:
            try:
                json.dump({k: result[k] for k in ("content", "usage", "model")},
                          open(cp, "w", encoding="utf-8"))
            except Exception:
                pass
        async with _lock:
            u = result.get("usage") or {}
            self.spend.add(model, u.get("prompt_tokens", 0), u.get("completion_tokens", 0))
            self._logf.write(json.dumps({
                "ts": time.time(), "model": model, "max_tokens": max_tokens,
                "temperature": temperature, "meta": meta,
                "usage": u, "error": result.get("error"),
                "content_preview": (result.get("content") or "")[:200],
            }, ensure_ascii=False) + "\n")
            self._since_flush += 1
            if self._since_flush >= 25:
                self._logf.flush()
                self.spend.flush()
                self._since_flush = 0
        return result

    async def _call_with_retry(self, model, messages, max_tokens, temperature):
        delay = 2.0
        temps_tried = set()
        for attempt in range(7):
            try:
                resp = await self.client.chat.completions.create(
                    model=model, messages=messages,
                    max_tokens=max_tokens, temperature=temperature)
                choice = resp.choices[0]
                content = choice.message.content or ""
                usage = {}
                if resp.usage:
                    usage = {"prompt_tokens": resp.usage.prompt_tokens,
                             "completion_tokens": resp.usage.completion_tokens}
                return {"content": content, "usage": usage, "error": None, "cached": False}
            except Exception as e:
                msg = str(e)
                code = getattr(e, "status_code", None)
                # temperature rejection -> fall back to 0.3 once
                if ("temperature" in msg.lower() and temperature == 0.0
                        and 0.3 not in temps_tried):
                    temps_tried.add(0.3)
                    temperature = C.TEMPERATURE_FALLBACK
                    continue
                retriable = (code in (429, 500, 502, 503, 504)
                             or "429" in msg or "timeout" in msg.lower()
                             or "rate" in msg.lower() or "overloaded" in msg.lower())
                if attempt < 6 and retriable:
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 60)
                    continue
                return {"content": "", "usage": {}, "error": f"{type(e).__name__}: {msg[:300]}",
                        "cached": False}
        return {"content": "", "usage": {}, "error": "max_retries", "cached": False}

    async def map(self, calls):
        """calls: list of dicts with keys model, messages, max_tokens, temperature?, meta?.
        Returns results in the same order."""
        async def one(c):
            return await self.chat(c["model"], c["messages"], c["max_tokens"],
                                   c.get("temperature"), c.get("meta"))
        return await asyncio.gather(*[one(c) for c in calls])

    def close(self):
        self._logf.flush()
        self._logf.close()
        self.spend.flush()


def run_calls(calls, concurrency=None, tag="run"):
    """Sync convenience wrapper. Returns (results, runner_spend_total)."""
    async def _go():
        r = Runner(concurrency=concurrency, tag=tag)
        try:
            res = await r.map(calls)
        finally:
            r.close()
        return res, r.spend.total
    return asyncio.run(_go())
