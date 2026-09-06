"""Detection metrics in the language of the inspection station.

The three numbers that decide whether this system is usable on a line:

    capture rate      TP / (TP + FN)   of the bad parts, how many we caught
    false alarm rate  FP / (FP + TN)   of the good parts, how many we wrongly stopped
    review load       (TP + FP) / N    of everything, how much lands in the queue

They are recall, FPR and flag rate under their engineering names. The plain
names are used throughout because these get read by people who run the line,
not only by people who build models.

Direction of improvement, stated once because it is easy to invert:
**capture rate should go UP, false alarm rate should go DOWN.** They trade off
against each other -- lowering the threshold catches more bad parts and stops
more good ones -- so a single operating point is always a compromise, and the
sweep exists to make that compromise visible rather than accidental.

RETAKE is excluded from every count. The photo was unusable, so nothing was
learned about the part; scoring it either way would be inventing an outcome.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Confusion:
    """Counts at one threshold, plus the metrics derived from them."""

    tp: int = 0
    fn: int = 0
    fp: int = 0
    tn: int = 0
    excluded: int = 0                 # RETAKE: photo unusable, not counted
    threshold: float = 0.0
    excluded_names: list[str] = field(default_factory=list)

    @property
    def n_scored(self) -> int:
        return self.tp + self.fn + self.fp + self.tn

    @property
    def n_ng(self) -> int:
        return self.tp + self.fn

    @property
    def n_ok(self) -> int:
        return self.fp + self.tn

    @property
    def capture_rate(self) -> float:
        """Of the bad parts, how many we caught. Higher is better."""
        return self.tp / self.n_ng if self.n_ng else float("nan")

    @property
    def false_alarm_rate(self) -> float:
        """Of the good parts, how many we wrongly stopped. Lower is better."""
        return self.fp / self.n_ok if self.n_ok else float("nan")

    @property
    def review_load(self) -> float:
        """Of everything, how much lands in the operator's queue."""
        n = self.n_scored
        return (self.tp + self.fp) / n if n else float("nan")

    @property
    def precision(self) -> float:
        """Of what we flagged, how much was genuinely bad."""
        flagged = self.tp + self.fp
        return self.tp / flagged if flagged else float("nan")

    @property
    def accuracy(self) -> float:
        n = self.n_scored
        return (self.tp + self.tn) / n if n else float("nan")

    def as_rows(self) -> list[dict]:
        """The three headline metrics, ready for a UI table."""
        return [
            {"Metric": "Capture rate",
             "Engineering name": "Recall / sensitivity / TPR",
             "Formula": "TP / (TP + FN)",
             "Value": _pct(self.capture_rate),
             "Counts": f"{self.tp} of {self.n_ng} NG",
             "Want": "HIGHER"},
            {"Metric": "False alarm rate",
             "Engineering name": "FPR / 1 - specificity",
             "Formula": "FP / (FP + TN)",
             "Value": _pct(self.false_alarm_rate),
             "Counts": f"{self.fp} of {self.n_ok} OK",
             "Want": "LOWER"},
            {"Metric": "Review load",
             "Engineering name": "Flag rate / positive rate",
             "Formula": "(TP + FP) / total",
             "Value": _pct(self.review_load),
             "Counts": f"{self.tp + self.fp} of {self.n_scored}",
             "Want": "LOWER"},
        ]


def _pct(x: float) -> str:
    return "n/a" if not np.isfinite(x) else f"{100 * x:.1f}%"


def confusion_at(scores, labels, threshold: float,
                 excluded_names: list[str] | None = None) -> Confusion:
    """Confusion counts for `score > threshold` meaning "flagged".

    `scores` and `labels` must already exclude RETAKE images; `excluded_names`
    is only carried through for reporting.
    """
    s = np.asarray(scores, float)
    y = np.asarray(labels, int)
    keep = np.isfinite(s)
    s, y = s[keep], y[keep]

    flagged = s > threshold
    return Confusion(
        tp=int(np.sum(flagged & (y == 1))),
        fn=int(np.sum(~flagged & (y == 1))),
        fp=int(np.sum(flagged & (y == 0))),
        tn=int(np.sum(~flagged & (y == 0))),
        excluded=len(excluded_names or []),
        threshold=float(threshold),
        excluded_names=list(excluded_names or []),
    )


def sweep(scores, labels, n: int = 300) -> list[dict]:
    """Metrics across the full range of thresholds.

    The grid spans the observed scores with a margin, so both ends (flag
    everything / flag nothing) are represented and the curve does not appear
    to stop mid-air.
    """
    s = np.asarray(scores, float)
    y = np.asarray(labels, int)
    keep = np.isfinite(s)
    s, y = s[keep], y[keep]
    if len(s) == 0:
        return []

    lo, hi = float(s.min()), float(s.max())
    span = max(hi - lo, 1e-6)
    grid = np.linspace(lo - 0.05 * span, hi + 0.05 * span, n)

    out = []
    for t in grid:
        c = confusion_at(s, y, float(t))
        out.append({
            "threshold": float(t),
            "capture_rate": c.capture_rate,
            "false_alarm_rate": c.false_alarm_rate,
            "review_load": c.review_load,
            "tp": c.tp, "fn": c.fn, "fp": c.fp, "tn": c.tn,
        })
    return out


def suggest_threshold(scores, labels, min_capture: float = 1.0):
    """Lowest false-alarm threshold that still meets a capture-rate target.

    Defaults to 100% capture because on a safety-relevant part a missed defect
    usually costs far more than a false alarm. Returns None when no threshold
    reaches the target -- which is itself the answer: the distributions
    overlap and no operating point separates them.
    """
    rows = sweep(scores, labels)
    ok = [r for r in rows if np.isfinite(r["capture_rate"])
          and r["capture_rate"] >= min_capture]
    if not ok:
        return None
    # Among thresholds meeting the target, the fewest false alarms; ties
    # broken by the higher threshold, which leaves more headroom.
    return min(ok, key=lambda r: (r["false_alarm_rate"], -r["threshold"]))


def roc(scores, labels) -> tuple[list[float], list[float], float]:
    """ROC points and AUROC, computed from the same sweep."""
    rows = sweep(scores, labels)
    if not rows:
        return [], [], float("nan")
    fpr = [r["false_alarm_rate"] for r in rows]
    tpr = [r["capture_rate"] for r in rows]
    order = np.argsort(fpr)
    x = np.asarray(fpr, float)[order]
    y = np.asarray(tpr, float)[order]
    good = np.isfinite(x) & np.isfinite(y)
    # np.trapezoid is NumPy 2.0+; np.trapz is the 1.x name and is deprecated
    # in 2.x. Support both so the demo runs on either.
    trap = getattr(np, "trapezoid", None) or np.trapz
    auc = float(trap(y[good], x[good])) if good.sum() > 1 else float("nan")
    return list(x[good]), list(y[good]), abs(auc)


def separation(scores, labels) -> float:
    """Standardised gap between the OK and NG score distributions.

    A large AUROC with a small separation means the ranking is right but the
    margin is thin -- worth knowing before trusting a threshold.
    """
    s = np.asarray(scores, float)
    y = np.asarray(labels, int)
    ok, ng = s[(y == 0) & np.isfinite(s)], s[(y == 1) & np.isfinite(s)]
    if len(ok) < 2 or len(ng) < 1:
        return float("nan")
    var_ng = ng.var(ddof=1) if len(ng) > 1 else 0.0
    pooled = np.sqrt((ok.var(ddof=1) + var_ng) / (2 if len(ng) > 1 else 1))
    return float("inf") if pooled <= 0 else float((ng.mean() - ok.mean()) / pooled)
