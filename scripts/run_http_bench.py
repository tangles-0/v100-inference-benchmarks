#!/usr/bin/env python3
"""Run the canonical completion payloads against an OpenAI-compatible server."""

import concurrent.futures
import json
import os
from pathlib import Path
import statistics
import time
import urllib.request

ROOT = Path(__file__).parent
URL = os.getenv("BENCH_URL", "http://127.0.0.1:8010/v1/completions")
MODEL = os.getenv("BENCH_MODEL", "qwen3.8-27b")


def request(payload: dict) -> tuple[int, float]:
    payload = dict(payload)
    payload["model"] = MODEL
    started = time.perf_counter()
    req = urllib.request.Request(
        URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as response:
        result = json.load(response)
    elapsed = time.perf_counter() - started
    return result["usage"]["completion_tokens"], elapsed


serial_template = json.loads((ROOT / "payloads/serial-256.json").read_text())
request(serial_template)  # unreported warm-up
serial: list[tuple[int, float]] = []
for run in range(5):
    payload = dict(serial_template)
    payload["prompt"] = payload["prompt"].replace("case 0", f"case {run}")
    payload["seed"] = 4242 + run
    tokens, elapsed = request(payload)
    serial.append((tokens, elapsed))
    print(f"serial run={run} tokens={tokens} elapsed={elapsed:.4f}s")

serial_rates = [tokens / elapsed for tokens, elapsed in serial]
print(
    "serial summary "
    f"aggregate={sum(t for t, _ in serial) / sum(s for _, s in serial):.2f} "
    f"median={statistics.median(serial_rates):.2f} tok/s"
)

batch_templates = [
    json.loads(line)
    for line in (ROOT / "payloads/batch-4x256.jsonl").read_text().splitlines()
]
batch_results: list[tuple[int, float]] = []
for wave in range(4):
    payloads = []
    for slot, template in enumerate(batch_templates):
        payload = dict(template)
        payload["prompt"] = payload["prompt"].replace(
            f"benchmark {slot}", f"benchmark {wave * 4 + slot}"
        )
        payload["seed"] = 7000 + wave * 4 + slot
        payloads.append(payload)
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        responses = list(executor.map(request, payloads))
    elapsed = time.perf_counter() - started
    tokens = sum(tokens for tokens, _ in responses)
    print(
        f"batch wave={wave} tokens={tokens} elapsed={elapsed:.4f}s "
        f"aggregate={tokens / elapsed:.2f} tok/s"
    )
    if wave:
        batch_results.append((tokens, elapsed))

print(
    "batch summary "
    f"aggregate={sum(t for t, _ in batch_results) / sum(s for _, s in batch_results):.2f} "
    "tok/s (three measured waves)"
)
