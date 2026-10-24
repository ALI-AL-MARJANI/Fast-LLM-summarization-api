**Engine:** vLLM 0.31.0, torch 2.13.0+cu130 (CUDA 13.0), `--dtype=half --max-model-len=8192 --gpu-memory-utilization=0.90`, repository commit e5c41a4  
**Model:** Qwen/Qwen3-4B-Instruct-2507  
**Hardware:** Google Colab, NVIDIA Tesla T4 (15360 MiB), driver 580.82.07  
**Run:** 2026-10-24 15:28 UTC, 64 streamed requests per level, max_tokens=128

| Concurrency | Requests | Errors | p50 TTFT (ms) | p95 TTFT (ms) | p50 latency (ms) | p95 latency (ms) | Requests/s | Output tokens/s |
|---|---|---|---|---|---|---|---|---|
| 1 | 64 | 0 | 83 | 89 | 2536 | 2763 | 0.42 | 26.3 |
| 4 | 64 | 0 | 131 | 2849 | 2959 | 5010 | 1.33 | 84.1 |
| 16 | 64 | 0 | 187 | 7940 | 4027 | 14688 | 2.43 | 153.6 |
| 32 | 64 | 0 | 309 | 2996 | 5565 | 8326 | 4.77 | 302.1 |
