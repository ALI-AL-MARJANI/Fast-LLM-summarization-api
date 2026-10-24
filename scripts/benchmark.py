"""Async load test for the summarization gateway.

Sends a fixed number of streamed requests at several concurrency levels and
prints a Markdown report: model, hardware, then a table with time to first
token (TTFT), end-to-end latency (p50/p95), requests/s, output tokens/s and
errors. Progress goes to stderr, so stdout can be pasted as is.

Usage (ideally on the server itself, to exclude internet latency):
    # GPU (default concurrency levels: 1 4 16 32)
    docker compose exec -T gateway python - --url http://localhost:8080 \
        --hardware "NVIDIA T4 16 GB" < scripts/benchmark.py
    # CPU (2 physical cores: use lower concurrency levels)
    docker compose exec -T gateway python - --url http://localhost:8080 \
        --hardware "EC2 m7i.xlarge" --concurrency 1 2 4 8 < scripts/benchmark.py
or from any machine with httpx installed:
    API_KEY=... python scripts/benchmark.py --url http://<host>:8080 --hardware "..."
"""

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import httpx

# Fixed inputs so that runs are comparable across models and configurations.
TEXTS = [
    "Continuous integration is a software practice in which developers merge their "
    "changes into a shared branch several times a day. Each merge triggers an automated "
    "build and a test suite, so that integration problems are detected within minutes "
    "instead of weeks. Teams that adopt it usually pair it with code review, short-lived "
    "branches and fast feedback loops. The main costs are the initial effort to write "
    "reliable tests and the infrastructure needed to run them on every change.",
    "Heat pumps move heat instead of producing it. In winter, a heat pump extracts heat "
    "from the outside air or the ground and releases it indoors; in summer, the cycle is "
    "reversed to cool the building. Because moving heat requires less energy than "
    "generating it, a heat pump can deliver three to four units of heat per unit of "
    "electricity. Their efficiency drops in very cold climates, and installation costs "
    "remain higher than those of gas boilers, although subsidies often reduce the gap.",
    "The city council approved a plan to convert two central streets into pedestrian "
    "zones starting next spring. Supporters argue that the change will reduce noise and "
    "pollution and attract more visitors to local shops. Several shop owners, however, "
    "fear that deliveries will become harder and that customers who drive will go "
    "elsewhere. The council promised dedicated delivery hours, new bicycle parking and a "
    "review of the measure after one year, based on footfall and air-quality data.",
]


@dataclass
class Result:
    ok: bool
    latency_s: float
    ttft_s: float = 0.0
    completion_tokens: int = 0
    model: str = ""


async def one_request(client: httpx.AsyncClient, text: str, max_tokens: int) -> Result:
    """One streamed summary: TTFT at the first `delta` event, latency at `done`."""
    payload = {"text": text, "style": "short", "max_tokens": max_tokens, "stream": True}
    start = time.perf_counter()
    ttft = None
    try:
        async with client.stream("POST", "/v1/summarize", json=payload) as response:
            if response.status_code != 200:
                return Result(ok=False, latency_s=time.perf_counter() - start)
            event = ""
            async for line in response.aiter_lines():
                if line.startswith("event: "):
                    event = line.removeprefix("event: ")
                    continue
                if not line.startswith("data: "):
                    continue  # blank separator lines between events
                if event == "delta" and ttft is None:
                    ttft = time.perf_counter() - start
                elif event == "done":
                    latency = time.perf_counter() - start
                    done = json.loads(line.removeprefix("data: "))
                    return Result(
                        True,
                        latency,
                        ttft if ttft is not None else latency,
                        done["usage"]["completion_tokens"],
                        done["model"],
                    )
                elif event == "error":
                    break
        # In-band `error` event, or stream closed without a `done` event.
        return Result(ok=False, latency_s=time.perf_counter() - start)
    except httpx.HTTPError:
        return Result(ok=False, latency_s=time.perf_counter() - start)


async def run_level(client: httpx.AsyncClient, concurrency: int, n: int, max_tokens: int):
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded(i: int) -> Result:
        async with semaphore:
            return await one_request(client, TEXTS[i % len(TEXTS)], max_tokens)

    start = time.perf_counter()
    results = await asyncio.gather(*(bounded(i) for i in range(n)))
    return results, time.perf_counter() - start


def percentile(sorted_values: list[float], q: float) -> float:
    """Nearest-rank percentile on an already sorted list."""
    if not sorted_values:
        return float("nan")
    index = max(0, min(len(sorted_values) - 1, round(q * len(sorted_values)) - 1))
    return sorted_values[index]


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://localhost:8080")
    parser.add_argument("--api-key", default=os.environ.get("API_KEY", ""))
    parser.add_argument("--hardware", default="unspecified",
                        help='server hardware, shown in the report (e.g. "EC2 m7i.xlarge")')
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 16, 32])
    parser.add_argument("--requests", type=int, default=64, help="requests per level")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=300,
                        help="client timeout in seconds (keep it above the gateway's)")
    args = parser.parse_args()

    limits = httpx.Limits(max_connections=max(args.concurrency))
    models: set[str] = set()
    async with httpx.AsyncClient(
        base_url=args.url, headers={"X-API-Key": args.api_key},
        timeout=args.timeout, limits=limits,
    ) as client:
        await run_level(client, 1, 2, args.max_tokens)  # warm-up, not reported
        rows = []
        for concurrency in args.concurrency:
            results, wall_s = await run_level(client, concurrency, args.requests, args.max_tokens)
            ok = [r for r in results if r.ok]
            models.update(r.model for r in ok)
            ttfts = sorted(r.ttft_s * 1000 for r in ok)
            latencies = sorted(r.latency_s * 1000 for r in ok)
            tokens = sum(r.completion_tokens for r in ok)
            print(f"concurrency {concurrency}: {len(ok)}/{len(results)} ok in {wall_s:.0f} s",
                  file=sys.stderr, flush=True)
            rows.append(
                f"| {concurrency} | {len(results)} | {len(results) - len(ok)} "
                f"| {percentile(ttfts, 0.50):.0f} | {percentile(ttfts, 0.95):.0f} "
                f"| {percentile(latencies, 0.50):.0f} | {percentile(latencies, 0.95):.0f} "
                f"| {len(ok) / wall_s:.2f} | {tokens / wall_s:.1f} |"
            )

    date = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"**Model:** {', '.join(sorted(models)) or 'unknown (no successful request)'}  ")
    print(f"**Hardware:** {args.hardware}  ")
    print(f"**Run:** {date}, {args.requests} streamed requests per level, "
          f"max_tokens={args.max_tokens}")
    print()
    print("| Concurrency | Requests | Errors | p50 TTFT (ms) | p95 TTFT (ms) "
          "| p50 latency (ms) | p95 latency (ms) | Requests/s | Output tokens/s |")
    print("|---|---|---|---|---|---|---|---|---|")
    print("\n".join(rows))


if __name__ == "__main__":
    asyncio.run(main())
