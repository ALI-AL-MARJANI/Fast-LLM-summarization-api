"""Draws assets/how_it_works.png, the overview figure of the README.

Numbers come from benchmarks/ (Qwen2.5-1.5B-Instruct, concurrency 1).
Usage: pip install matplotlib && python scripts/make_figure.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parent.parent / "assets" / "how_it_works.png"

BG, CARD, INK, MUTED, ARROW = "#fbfbf9", "#f1f0ed", "#111111", "#555555", "#888888"
STEPS = [
    ("#2a7ad6", "1  Client", "curl, an app, the benchmark",
     'POST /v1/summarize\nX-API-Key: ****\n{"text": "...", "stream": true}',
     "→ sends a text and an API key\n   over HTTP"),
    ("#ec6a37", "2  FastAPI gateway", "the only public door, :8080",
     "checks the API key\nvalidates the input\nbuilds the prompt",
     "→ 401 bad key, 422 bad input,\n   503 when the model is down"),
    ("#00873b", "3  vLLM", "the model server, private :8000",
     "batches all waiting requests\ngenerates token by token\nstreams tokens back",
     "→ never exposed to the internet"),
]
HARDWARE = [
    ("EC2 m7i.xlarge · 2 CPU cores", "deployed on AWS", "4.8 tokens/s", "13.8 s per summary"),
    ("NVIDIA T4 GPU", "measured on Google Colab", "65.4 tokens/s", "1.0 s per summary  ·  13.6× faster"),
]


def box(ax, x, y, w, h, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=8",
                                facecolor=color, edgecolor="none"))


def main() -> None:
    fig = plt.figure(figsize=(20, 10.5), dpi=120, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 2000)
    ax.set_ylim(1050, 0)
    ax.axis("off")

    ax.text(58, 55, "How a summary is made, and how fast", fontsize=26, weight="bold", color=INK, va="center")
    ax.text(58, 120, "One request through the stack. Everything runs in Docker Compose on one EC2 instance.",
            fontsize=15, color=MUTED, va="center")

    width, gap = 600, 50
    for i, (color, title, subtitle, card, note) in enumerate(STEPS):
        x = 58 + i * (width + gap)
        box(ax, x, 180, width, 125, color)
        ax.text(x + 30, 225, title, fontsize=19, weight="bold", color="white", va="center")
        ax.text(x + 30, 272, subtitle, fontsize=14.5, color="white", va="center")
        box(ax, x, 330, width, 170, CARD)
        ax.text(x + 25, 415, card, fontsize=14, family="monospace", color=INK, va="center", linespacing=1.5)
        ax.text(x + 25, 555, note, fontsize=14.5, color=INK, va="center", linespacing=1.5)
        if i < len(STEPS) - 1:
            ax.annotate("", xy=(x + width + gap - 8, 242), xytext=(x + width + 8, 242),
                        arrowprops={"arrowstyle": "-|>", "color": ARROW, "lw": 2.5})

    ax.text(58, 655, "Where vLLM runs: same model (Qwen2.5-1.5B), one request at a time",
            fontsize=17, weight="bold", color=INK, va="center")
    width = 925
    for i, (name, where, speed, latency) in enumerate(HARDWARE):
        x = 58 + i * (width + 50)
        box(ax, x, 700, width, 230, CARD)
        ax.text(x + 30, 750, name, fontsize=18, weight="bold", color=INK, va="center")
        ax.text(x + 30, 795, where, fontsize=14.5, color=MUTED, va="center")
        ax.text(x + 30, 860, speed, fontsize=30, weight="bold", color=STEPS[2][0], va="center")
        ax.text(x + 30, 905, latency, fontsize=15, color=INK, va="center")

    ax.text(58, 1000, "Each new token means reading all the model's weights; in these runs, "
            "the GPU read them about 13× faster than 2 CPU cores.", fontsize=15, color=MUTED, va="center")

    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, facecolor=BG)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
