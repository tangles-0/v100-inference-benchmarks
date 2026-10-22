# V100 inference benchmarks

Reproducible inference measurements and implementation notes for models
tested on NVIDIA Tesla V100 hardware.

## Models

| Model | Hardware | Results |
|---|---|---|
| Qwen3.8-27B | 4 x V100-SXM2-16GB | [Results and analysis](models/qwen3.8-27b/) |

Each model directory owns its payloads, raw client measurements, stack
versions, and interpretation. Shared benchmark clients live in `scripts/`.

## Repository layout

```text
models/
  qwen3.8-27b/
    README.md
    payloads/
    raw/
scripts/
  run_http_bench.py
tools/                 # local research checkouts; ignored
local-artifacts/       # server logs and diagnostics; ignored
```

Throughput figures are completion tokens divided by client wall time unless
the model report explicitly says otherwise. They should not be compared
directly with server-side pure-decode figures.
