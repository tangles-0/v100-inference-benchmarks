# Benchmark payloads

These payloads are the canonical cross-stack requests for this host.

- `serial-256.json`: fixed-length single-stream decode. Run one unreported
  warm-up, then five measured requests. For measured run `n`, replace the
  prompt's case number with `n` and use seed `4242+n`.
- `batch-4x256.jsonl`: submit all four lines concurrently. Run one unreported
  warm-up wave, then three measured waves. Later waves increment each prompt
  number and seed by four while retaining all other fields.
- `natural-96.json`: deterministic sanity/trajectory comparison. Normal EOS
  remains enabled.

Use the model name shown where possible. If a runtime requires a different
served name or API schema, change only that routing field and document the
translation in `../results.md`.

Reported HTTP throughput is total completion tokens divided by client wall
time. It includes scheduling and decode but, for these short prompts, only a
small prefill. This is deliberately different from 1Cat's server-side
"pure decode" metric and should not be compared to that number without saying
so.
