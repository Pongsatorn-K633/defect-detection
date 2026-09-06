"""Charts for the detection metrics: confusion matrix, sweep, ROC, scores.

All rendered with matplotlib to PNG so the Gradio UI can show them as images,
and so the same figures can be dropped into a report.
"""
from __future__ import annotations

import numpy as np

_OK = "#2e7d32"       # good part
_NG = "#c62828"       # defective part
_HIT = "#2e7d32"      # correct decision
_MISS = "#c62828"     # wrong decision
_GREY = "#9e9e9e"


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _fig_to_rgb(fig):
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    _plt().close(fig)
    return buf


def confusion_matrix(c, title: str = "") -> np.ndarray:
    """2x2 matrix, sized and coloured by the actual counts.

    Cell colour tracks whether the decision was right, not how large the count
    is: a big TN and a big FP are both large numbers but only one is good, and
    colouring by magnitude alone would hide that.
    """
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.4, 4.4))

    cells = [[(c.tn, "TN", "correct pass", _HIT), (c.fp, "FP", "false alarm", _MISS)],
             [(c.fn, "FN", "MISSED defect", _MISS), (c.tp, "TP", "caught", _HIT)]]

    biggest = max(c.tp, c.fn, c.fp, c.tn, 1)
    for r in range(2):
        for col in range(2):
            n, code, meaning, colour = cells[r][col]
            # Alpha carries magnitude, hue carries right/wrong.
            ax.add_patch(plt.Rectangle((col, 1 - r), 1, 1, facecolor=colour,
                                       alpha=0.14 + 0.5 * (n / biggest),
                                       edgecolor="white", linewidth=3))
            ax.text(col + 0.5, 1 - r + 0.60, f"{n}", ha="center", va="center",
                    fontsize=25, fontweight="bold", color="#212121")
            ax.text(col + 0.5, 1 - r + 0.34, code, ha="center", va="center",
                    fontsize=11, fontweight="bold", color=colour)
            ax.text(col + 0.5, 1 - r + 0.16, meaning, ha="center", va="center",
                    fontsize=9, color="#555555")

    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5])
    ax.set_yticks([0.5, 1.5])
    ax.set_xticklabels(["PASS\n(system says OK)", "FAIL\n(system flags)"], fontsize=10)
    ax.set_yticklabels(["NG\n(truly bad)", "OK\n(truly good)"], fontsize=10)
    ax.set_xlabel("system decision", fontsize=10, labelpad=8)
    ax.set_ylabel("ground truth", fontsize=10, labelpad=8)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)

    head = title or f"Confusion matrix at threshold {c.threshold:.2f}"
    if c.excluded:
        # Second line rather than appended: a long single title gets clipped.
        head += f"\n{c.excluded} RETAKE excluded (photo unusable)"
    ax.set_title(head, fontsize=10.5, pad=10)
    fig.tight_layout()
    return _fig_to_rgb(fig)


def sweep_chart(rows, current: float | None = None,
                suggested: float | None = None) -> np.ndarray:
    """Capture rate and false alarm rate against threshold.

    The whole trade-off on one axis: moving the threshold left catches more
    defects and stops more good parts. There is no setting that improves both.
    """
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    if not rows:
        ax.text(0.5, 0.5, "no data", ha="center", va="center")
        return _fig_to_rgb(fig)

    t = [r["threshold"] for r in rows]
    ax.plot(t, [100 * r["capture_rate"] for r in rows], color=_HIT, lw=2.2,
            label="capture rate  (want HIGH)")
    ax.plot(t, [100 * r["false_alarm_rate"] for r in rows], color=_MISS, lw=2.2,
            label="false alarm rate  (want LOW)")
    ax.plot(t, [100 * r["review_load"] for r in rows], color=_GREY, lw=1.4,
            ls="--", label="review load")

    if suggested is not None:
        ax.axvline(suggested, color="#1565c0", ls=":", lw=1.8,
                   label=f"suggested {suggested:.2f}")
    if current is not None:
        ax.axvline(current, color="#212121", lw=1.8,
                   label=f"current {current:.2f}")

    ax.set_xlabel("threshold")
    ax.set_ylabel("percent")
    ax.set_ylim(-3, 103)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="center right")
    ax.set_title("Moving the threshold trades one against the other", fontsize=11)
    fig.tight_layout()
    return _fig_to_rgb(fig)


def score_distribution(scores, labels, names=None,
                       threshold: float | None = None) -> np.ndarray:
    """Every part's score, split by true label.

    This is the plot that shows whether the two groups genuinely separate or
    merely overlap less. A summary metric can look healthy while the nearest
    OK and NG parts sit almost on top of each other.
    """
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    s = np.asarray(scores, float)
    y = np.asarray(labels, int)

    rng = np.random.default_rng(0)          # deterministic jitter
    for lab, colour, name, row in ((0, _OK, "OK", 1.0), (1, _NG, "NG", 0.0)):
        sel = (y == lab) & np.isfinite(s)
        if not sel.any():
            continue
        ax.scatter(s[sel], row + rng.uniform(-0.11, 0.11, sel.sum()),
                   s=110, color=colour, alpha=0.75, edgecolor="white",
                   linewidth=1.4, label=f"{name}  (n={int(sel.sum())})", zorder=3)

    if threshold is not None:
        ax.axvline(threshold, color="#212121", lw=1.8, zorder=2)
        ax.text(threshold, 1.42, f" threshold {threshold:.2f}", fontsize=9,
                va="top", color="#212121")

    # Name the parts nearest the boundary from each side, and state the gap
    # between them. That gap is what decides whether the threshold survives
    # the next batch -- a healthy summary metric can sit on top of a margin of
    # almost nothing.
    if names is not None and threshold is not None:
        edge = {}
        for lab, row, pick in ((0, 1.0, np.argmax), (1, 0.0, np.argmin)):
            sel = np.where((y == lab) & np.isfinite(s))[0]
            if len(sel):
                i = sel[pick(s[sel])]
                edge[lab] = s[i]
                ax.annotate(f"{names[i]}\n{s[i]:.1f}", (s[i], row), fontsize=8,
                            xytext=(0, -26 if lab == 0 else 22),
                            textcoords="offset points", ha="center",
                            color="#555555")
        if 0 in edge and 1 in edge:
            ax.set_title("Score of every part, by true label      "
                         f"closest OK {edge[0]:.1f}  |  closest NG "
                         f"{edge[1]:.1f}   (gap {edge[1] - edge[0]:+.1f})",
                         fontsize=10)

    ax.set_yticks([0, 1])
    ax.set_yticklabels(["NG", "OK"])
    ax.set_ylim(-0.6, 1.6)
    ax.set_xlabel("anomaly score")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(fontsize=8, loc="upper right")
    if not ax.get_title():
        ax.set_title("Score of every part, by true label", fontsize=11)
    fig.tight_layout()
    return _fig_to_rgb(fig)


def roc_chart(fpr, tpr, auc: float) -> np.ndarray:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.plot([0, 1], [0, 1], ls="--", color=_GREY, lw=1.2, label="chance")
    if fpr:
        ax.plot(fpr, tpr, color="#1565c0", lw=2.4,
                label=f"AUROC {auc:.3f}" if np.isfinite(auc) else "AUROC n/a")
    ax.set_xlabel("false alarm rate")
    ax.set_ylabel("capture rate")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_title("ROC", fontsize=11)
    fig.tight_layout()
    return _fig_to_rgb(fig)
