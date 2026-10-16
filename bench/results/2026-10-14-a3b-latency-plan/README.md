# SSE latency benchmark preparation

`tools/bench_a3b_latency.py --prepare-only` was run against port 8096. It called model listing and the read-only `/v1/messages/count_tokens` endpoint. It did not run model inference, reset model state, or measure throughput.

The three prepared inputs count exactly 32, 512 and 2,048 tokens through the server's chat-template/tokenizer path. Full messages are in `plan.json`. Each actual streamed run will additionally record its authoritative OpenAI usage count and whether it agrees with the count endpoint.

The measurement code was separately checked with a local synthetic SSE server: it sent role-only and heartbeat events before reasoning and visible text. The client recorded first response byte, parsed headers and first body independently; first-any text followed the reasoning event, first-visible followed the later content event. No synthetic throughput result is a model benchmark.

Example real runs, only after the intended engine is ready:

```powershell
# Retain the resident engine; initial expert cache is explicitly unknown.
.venv\Scripts\python.exe tools\bench_a3b_latency.py --port 8096 --out bench\results\latency-resident --repeats 3

# Three independent cold/hot pairs per prompt/output case; intentionally reloads the model.
.venv\Scripts\python.exe tools\bench_a3b_latency.py --port 8096 --out bench\results\latency-cold-hot --repeats 3 --cache-policy cold-hot-pairs
```

SSE chunks can contain several tokens, especially when MTP produces bursts. Client decode rate uses API completion-token count over the first/last nonempty delta interval and is reported separately from engine timing. Role events, comments, prompt progress, empty deltas and final usage do not qualify as first-token content. Reasoning-only output has no visible TTFT; the visible-only rate remains null when reasoning was emitted.
