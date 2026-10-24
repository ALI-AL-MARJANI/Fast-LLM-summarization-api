**Engine:** vLLM 0.31.0, torch 2.13.0+cu130 (CUDA 13.0), `--dtype=half --max-model-len=8192 --gpu-memory-utilization=0.90`, repository commit e5c41a4  
**Model:** Qwen/Qwen2.5-1.5B-Instruct  
**Hardware:** Google Colab, NVIDIA Tesla T4 (15360 MiB), driver 580.82.07  
**Run:** 2026-10-24 15:19 UTC, 64 streamed requests per level, max_tokens=128

| Concurrency | Requests | Errors | p50 TTFT (ms) | p95 TTFT (ms) | p50 latency (ms) | p95 latency (ms) | Requests/s | Output tokens/s |
|---|---|---|---|---|---|---|---|---|
| 1 | 64 | 0 | 37 | 40 | 1040 | 1290 | 0.96 | 65.4 |
| 4 | 64 | 0 | 64 | 3049 | 1151 | 3818 | 2.97 | 202.0 |
| 16 | 64 | 0 | 106 | 8136 | 1622 | 12720 | 3.68 | 251.4 |
| 32 | 64 | 0 | 367 | 3206 | 2419 | 5407 | 8.50 | 583.8 |
