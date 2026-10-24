**Model:** Qwen/Qwen2.5-1.5B-Instruct  
**Hardware:** EC2 m7i.xlarge (Xeon Platinum 8488C, 2 cores / 4 threads, 16 GB)  
**Run:** 2026-10-24 11:34 UTC, 32 streamed requests per level, max_tokens=128

| Concurrency | Requests | Errors | p50 TTFT (ms) | p95 TTFT (ms) | p50 latency (ms) | p95 latency (ms) | Requests/s | Output tokens/s |
|---|---|---|---|---|---|---|---|---|
| 1 | 32 | 0 | 299 | 326 | 13808 | 16040 | 0.07 | 4.8 |
| 2 | 32 | 0 | 480 | 520 | 14418 | 17873 | 0.14 | 9.3 |
| 4 | 32 | 0 | 497 | 826 | 15542 | 18235 | 0.25 | 17.4 |
| 8 | 32 | 0 | 531 | 1321 | 15522 | 19626 | 0.47 | 32.4 |
