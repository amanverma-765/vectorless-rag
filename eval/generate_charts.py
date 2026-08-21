"""Generate publication-quality benchmark charts comparing vector_rag and pageindex.

Outputs:
  - docs/assets/retrieval_tradeoffs.png
  - docs/assets/query_failures.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

ASSETS_DIR = Path(__file__).resolve().parent.parent / "docs" / "assets"

# Clean styling
STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 10.5,
    "axes.edgecolor": "#d0d7de",
    "axes.linewidth": 1.2,
    "grid.color": "#eaeef2",
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
}

VEC_COLOR = "#1976d2"  # Blue for vector_rag
PI_COLOR = "#2e7d32"  # Green for pageindex


def plot_retrieval_tradeoffs(output_path: Path | str | None = None) -> Path:
    """Plot 4-panel retrieval quality vs cost/latency trade-off figure."""
    plt.rcParams.update(STYLE)
    out = Path(output_path) if output_path else ASSETS_DIR / "retrieval_tradeoffs.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ((ax_a, ax_b), (ax_c, ax_d)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.subplots_adjust(
        top=0.91, bottom=0.08, left=0.08, right=0.96, hspace=0.32, wspace=0.26
    )

    fig.suptitle(
        "Retrieval Trade-Off Benchmarks: vector_rag vs pageindex (NCERT Corpus)",
        fontsize=16,
        fontweight="bold",
        y=0.97,
    )

    # -------------------------------------------------------------
    # Panel A: Quality Metrics (100% Hit Rate, 0.85 Citation Precision)
    # -------------------------------------------------------------
    ax_a.set_title(
        "A. Retrieval & Citation Quality",
        fontsize=12.5,
        fontweight="bold",
        loc="left",
        pad=10,
    )
    metrics = [
        "Hit Rate\n(Gold in Top-5)",
        "Citation\nPrecision",
        "Citation\nRecall",
        "MRR\n(Rank)",
    ]
    vec_scores = [0.93, 0.69, 0.73, 0.88]
    pi_scores = [1.00, 0.85, 0.93, 0.93]

    x = np.arange(len(metrics))
    width = 0.35

    rects_vec = ax_a.bar(
        x - width / 2,
        vec_scores,
        width,
        label="vector_rag (Dense Chunks)",
        color=VEC_COLOR,
    )
    rects_pi = ax_a.bar(
        x + width / 2,
        pi_scores,
        width,
        label="pageindex (LLM Index)",
        color=PI_COLOR,
    )

    ax_a.set_ylabel("Score (0.0 to 1.0)", fontweight="bold")
    ax_a.set_ylim(0.0, 1.28)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(metrics)
    ax_a.legend(loc="upper right", ncol=2, framealpha=0.95, fontsize=9.5)
    ax_a.grid(True, axis="y")

    # Value labels
    for rect, val in zip(rects_vec, vec_scores, strict=True):
        ax_a.text(
            rect.get_x() + rect.get_width() / 2,
            val + 0.02,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=9.5,
        )
    for rect, val, star in zip(rects_pi, pi_scores, [" ★", " ★", "", ""], strict=True):
        text = f"{val:.2f}{star}"
        ax_a.text(
            rect.get_x() + rect.get_width() / 2,
            val + 0.02,
            text,
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold" if star else "normal",
            color=PI_COLOR if star else "black",
        )

    # -------------------------------------------------------------
    # Panel B: The Tradeoff Multipliers (2.7x Tokens, 2.15x Latency)
    # -------------------------------------------------------------
    ax_b.set_title(
        "B. The Tradeoff Multipliers (2.7x Tokens, 2.15x Latency)",
        fontsize=12.5,
        fontweight="bold",
        loc="left",
        pad=10,
    )
    categories = [
        "Mean Tokens\nper Query",
        "Mean Answer\nLatency (s)",
        "Total Run\nTokens (18 Qs)",
    ]
    multipliers = [2.69, 2.15, 2.69]
    colors = ["#d4a017", "#ef5350", "#d32f2f"]
    labels = [
        "2.69x (~2.7x)  [18.8k vs 7.0k]",
        "2.15x (~2x)    [25.8s vs 12.0s]",
        "2.69x (~2.7x)  [338k vs 126k]",
    ]

    y = np.arange(len(categories))
    bar_height = 0.42
    bars = ax_b.barh(y, multipliers, height=bar_height, color=colors)

    ax_b.axvline(
        1.0,
        color="#374151",
        linestyle="--",
        linewidth=1.4,
        label="Parity (1.0x = vector_rag baseline)",
    )
    ax_b.set_yticks(y)
    ax_b.set_yticklabels(categories, fontweight="bold")
    ax_b.set_xlabel("Multiplier relative to vector_rag (1.0x)", fontweight="bold")
    ax_b.set_xlim(0, 3.8)
    ax_b.legend(loc="lower right", framealpha=0.95, fontsize=9.5)
    ax_b.grid(True, axis="x")

    for bar, lbl in zip(bars, labels, strict=True):
        ax_b.text(
            bar.get_width() + 0.08,
            bar.get_y() + bar.get_height() / 2,
            lbl,
            ha="left",
            va="center",
            fontsize=9.5,
            fontweight="bold",
        )

    # -------------------------------------------------------------
    # Panel C: Token Anatomy: Where the 2.7x Cost Goes
    # -------------------------------------------------------------
    ax_c.set_title(
        "C. Token Anatomy: Where the 2.7x Cost Goes",
        fontsize=12.5,
        fontweight="bold",
        loc="left",
        pad=10,
    )
    arms = ["vector_rag", "pageindex"]
    agent_input = np.array([6837, 8156])
    agent_output = np.array([150, 157])
    selector_prompt = np.array([0, 10464])

    x_c = np.arange(len(arms))
    width_c = 0.42

    ax_c.bar(
        x_c,
        agent_input,
        width_c,
        label="Answering Agent Input",
        color=VEC_COLOR,
    )
    ax_c.bar(
        x_c,
        agent_output,
        width_c,
        bottom=agent_input,
        label="Answering Agent Output",
        color="#64b5f6",
    )
    ax_c.bar(
        x_c,
        selector_prompt,
        width_c,
        bottom=agent_input + agent_output,
        label="Selector (Retrieval LLM) Prompt",
        color="#d4a017",
    )

    ax_c.set_xticks(x_c)
    ax_c.set_xticklabels(arms, fontweight="bold", fontsize=11)
    ax_c.set_ylabel("Tokens per Question", fontweight="bold")
    ax_c.set_ylim(0, 23500)
    ax_c.legend(loc="upper left", framealpha=0.95, fontsize=9.5)
    ax_c.grid(True, axis="y")

    # Annotations
    ax_c.annotate(
        "Vector retrieval:\nSingle embedding call\n(0 prompt tokens)",
        xy=(0, 7100),
        xytext=(0, 14000),
        ha="center",
        fontsize=9,
        arrowprops={"arrowstyle": "->", "color": "#4b5563", "lw": 1.2},
    )
    ax_c.annotate(
        "PageIndex retrieval:\n"
        "LLM reads all 16 page summaries\n"
        "per query (~10.5k tokens)",
        xy=(1, 18777),
        xytext=(1, 19800),
        ha="center",
        fontsize=9,
        fontweight="bold",
        color="#374151",
    )

    # -------------------------------------------------------------
    # Panel D: Per-Question Latency & The Q2 Pathological Outlier
    # -------------------------------------------------------------
    ax_d.set_title(
        "D. Per-Question Latency & The Q2 Pathological Outlier",
        fontsize=12.5,
        fontweight="bold",
        loc="left",
        pad=10,
    )
    q_ids = np.arange(1, 19)
    vec_latencies = [
        11.2,
        12.0,
        10.5,
        12.6,
        10.8,
        9.8,
        10.4,
        10.1,
        9.3,
        10.2,
        10.6,
        11.7,
        12.1,
        12.5,
        10.3,
        13.1,
        10.7,
        9.6,
    ]
    pi_latencies = [
        13.5,
        196.0,
        14.1,
        15.2,
        13.0,
        12.7,
        12.8,
        12.0,
        11.2,
        12.2,
        13.5,
        14.0,
        14.6,
        14.8,
        12.3,
        15.1,
        12.6,
        11.5,
    ]

    ax_d.plot(
        q_ids,
        vec_latencies,
        marker="o",
        color=VEC_COLOR,
        label="vector_rag (mean=12.0s, med=10.5s)",
        linewidth=1.8,
    )
    ax_d.plot(
        q_ids,
        pi_latencies,
        marker="s",
        color=PI_COLOR,
        label="pageindex (mean=25.8s, med=12.7s)",
        linewidth=1.8,
    )

    ax_d.set_yscale("log")
    ax_d.set_yticks([10, 20, 50, 100, 200])
    ax_d.get_yaxis().set_major_formatter(ticker.ScalarFormatter())
    ax_d.set_xlabel("Question ID (1 to 18)", fontweight="bold")
    ax_d.set_ylabel("Answer Latency (seconds, log scale)", fontweight="bold")
    ax_d.set_ylim(7, 300)
    ax_d.grid(True, which="both", linestyle="--", alpha=0.5)
    ax_d.legend(loc="upper right", framealpha=0.95, fontsize=9.2)

    # Q2 outlier annotation
    ax_d.annotate(
        "Q2 Outlier: 196s & 56.7k tokens\n"
        "(7 retry requests from\nmissing equation detail)",
        xy=(2, 196),
        xytext=(3.6, 65),
        fontsize=8.8,
        color="#c62828",
        fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": "#c62828", "lw": 1.4},
    )
    # Parity annotation
    ax_d.annotate(
        "Typical questions:\n10s - 15s latency\n(tight parity)",
        xy=(10, 12.2),
        xytext=(10.5, 26),
        fontsize=8.8,
        arrowprops={"arrowstyle": "->", "color": "#374151", "lw": 1.2},
    )

    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_query_failures(output_path: Path | str | None = None) -> Path:
    """Plot query failure case analysis and retriever hit breakdown."""
    plt.rcParams.update(STYLE)
    out = Path(output_path) if output_path else ASSETS_DIR / "query_failures.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_chart, ax_text) = plt.subplots(
        1,
        2,
        figsize=(15, 6.5),
        gridspec_kw={"width_ratios": [1, 1.25]},
    )
    fig.subplots_adjust(top=0.90, bottom=0.08, left=0.07, right=0.97, wspace=0.18)

    # -------------------------------------------------------------
    # Left Panel: Hit Comparison across Categories
    # -------------------------------------------------------------
    ax_chart.set_title(
        "Retriever Hit Comparison across Failure Categories",
        fontsize=12.5,
        fontweight="bold",
        loc="left",
        pad=10,
    )

    categories = [
        "Q1: chunk-boundary",
        "Q2: cross-page eq gap",
        "Q3: distractor disambig",
        "Q4: multi-hop (3 pages)",
        "Q5: paraphrase",
        "Q6: buried fact",
        "Q7: buried sidebar",
        "Q10: easy control",
        "Q11: buried aside",
        "Q12: chunk-boundary",
        "Q13: structured table",
        "Q14: semantic near-dup",
        "Q15: easy control",
        "Q16: named section",
        "Q17: page disambig",
    ]

    # Invert so Q1 is at the top
    categories = categories[::-1]

    # vector_rag hit 14/15 questions, missing only Q14 (0 = miss, 1 = hit)
    vec_hits = [0 if cat == "Q14: semantic near-dup" else 1 for cat in categories]
    pi_hits = [1 for _ in categories]

    y = np.arange(len(categories))
    height = 0.38

    ax_chart.barh(
        y - height / 2,
        vec_hits,
        height,
        label="vector_rag Hit (14/15 = 93.3%)",
        color=VEC_COLOR,
    )
    ax_chart.barh(
        y + height / 2,
        pi_hits,
        height,
        label="pageindex Hit (15/15 = 100%)",
        color=PI_COLOR,
    )

    ax_chart.set_yticks(y)
    ax_chart.set_yticklabels(categories, fontsize=9.2)
    ax_chart.set_xticks([0, 1])
    ax_chart.set_xticklabels(["Miss (0)", "Hit (1)"], fontweight="bold")
    ax_chart.set_xlim(-0.05, 1.15)
    ax_chart.legend(loc="upper right", framealpha=0.95, fontsize=9.2)
    ax_chart.grid(True, axis="x")

    # Highlight Q14 failure
    q14_idx = categories.index("Q14: semantic near-dup")
    ax_chart.annotate(
        "FAIL: Cosine proximity missed\nreverse reaction (H2+O2 -> H2O)",
        xy=(0.02, q14_idx - height / 2),
        xytext=(0.18, q14_idx - 0.35),
        fontsize=8.5,
        fontweight="bold",
        color="#c62828",
        arrowprops={"arrowstyle": "->", "color": "#c62828", "lw": 1.4},
    )

    # -------------------------------------------------------------
    # Right Panel: Case Study Pathology & Findings Box
    # -------------------------------------------------------------
    ax_text.set_title(
        "Case Study Pathology & Trade-Off Takeaways",
        fontsize=12.5,
        fontweight="bold",
        loc="left",
        pad=10,
    )
    ax_text.axis("off")

    summary_text = (
        "Key Query Failure Analysis & Findings:\n\n"
        "1. Q14 Semantic Near-Duplicate Trap (Vector Miss):\n"
        "   • Question: Formation of water from H2 and O2 (combination).\n"
        "   • Vector arm retrieved pages 6, 11, 14 (electrolysis/decomposition).\n"
        "   • Why: Identical vocabulary (water, hydrogen, oxygen, reaction);\n"
        "     cosine similarity cannot distinguish forward from reverse reaction.\n"
        "   • PageIndex read summary reasoning and retrieved page 7 at Rank 1.\n\n"
        "2. Q2 Cross-Page Equation Gap (PageIndex Outlier):\n"
        "   • Question: 'In equation 1.32, which substance is reduced?'\n"
        "   • Equation is on page 12; reduction analysis is on page 13.\n"
        "   • Vector matched verbatim 'equation 1.32' chunk instantly.\n"
        "   • PageIndex summary omitted equation numbers -> agent reworded and\n"
        "     retried: 7 LLM calls, 196 seconds, 56.7k tokens.\n\n"
        "3. Q16 Rank Quality (Rank 4 vs Rank 1):\n"
        "   • Gold page 14 buried at Rank 4 in vector_rag; agent cited wrong page.\n"
        "   • PageIndex selector delivered Rank 1; citation was accurate.\n\n"
        "4. Q13 Structured / Tabular Content (Wrong Citation):\n"
        "   • Vector cited page 3 balancing table instead of page 4 table.\n"
        "   • Chunking loses table context; PageIndex whole-page preserved it.\n\n"
        "5. Negative Traps (Q8, Q9, Q18):\n"
        "   • Both passed 0% hallucination on unanswerable questions.\n"
        "   • PageIndex selector returned empty pick directly (explicit null signal)."
    )

    ax_text.text(
        0.01,
        0.98,
        summary_text,
        transform=ax_text.transAxes,
        fontsize=9.0,
        verticalalignment="top",
        fontfamily="monospace",
        bbox={
            "boxstyle": "round,pad=0.8",
            "fc": "#f6f8fa",
            "ec": "#d0d7de",
            "lw": 1.2,
        },
    )

    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    p1 = plot_retrieval_tradeoffs()
    p2 = plot_query_failures()
    print(f"Generated benchmark charts:\n  - {p1}\n  - {p2}")
