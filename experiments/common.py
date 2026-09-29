"""Shared helpers for the experiment scripts."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inspector.config import GOLDEN_DIR, OUTPUT_DIR, TEST_DIR  # noqa: E402
from inspector.pipeline import IMAGE_SUFFIXES, load_image  # noqa: E402

IMAGE_EXTS = IMAGE_SUFFIXES


def gather(folder: Path, skip_sheets: bool = True) -> list[Path]:
    """Image files in a folder, excluding the IS_* inspection sheets."""
    if not folder.exists():
        return []
    return sorted(
        p for p in folder.iterdir()
        if p.suffix.lower() in IMAGE_EXTS
        and not (skip_sheets and p.name.upper().startswith("IS_"))
    )


def golden_paths(sku: str) -> list[Path]:
    return gather(GOLDEN_DIR / sku)


def test_paths(sku: str) -> list[Path]:
    return gather(TEST_DIR / sku)


def load_all(paths: list[Path]) -> list[np.ndarray]:
    return [load_image(p) for p in paths]


def out_dir(name: str, sku: str) -> Path:
    d = OUTPUT_DIR / name / sku
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_labels(sku: str) -> dict[str, int]:
    """Ground-truth labels for the test set, if the operator has supplied them.

    Format: `data/labels/<sku>.csv` with `image,label` rows where label is 0
    for a good part and 1 for a defective one. Without this file the scripts
    that need labels report score distributions only and say so, rather than
    inventing an accuracy number.
    """
    from inspector.config import DATA_DIR

    path = DATA_DIR / "labels" / f"{sku}.csv"
    if not path.exists():
        return {}
    labels = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2 or parts[1] in ("", "?"):
            continue
        try:
            labels[parts[0]] = int(parts[1])
        except ValueError:
            # The header row, or a stray comment -- skip rather than crash, so
            # the file can carry explanatory notes about how it was produced.
            continue
    return labels


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC via the rank statistic; ties handled by average ranks."""
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, int)
    keep = np.isfinite(scores)
    scores, labels = scores[keep], labels[keep]
    n_pos, n_neg = int((labels == 1).sum()), int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # Average ranks within tied score groups.
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) /
                 (n_pos * n_neg))


def separation(scores: np.ndarray, labels: np.ndarray) -> float:
    """Standardised gap between the good and defective score distributions."""
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, int)
    good, bad = scores[labels == 0], scores[labels == 1]
    good, bad = good[np.isfinite(good)], bad[np.isfinite(bad)]
    if len(good) < 2 or len(bad) < 1:
        return float("nan")
    pooled = np.sqrt((good.var(ddof=1) + (bad.var(ddof=1) if len(bad) > 1 else 0))
                     / (2 if len(bad) > 1 else 1))
    if pooled <= 0:
        return float("inf")
    return float((bad.mean() - good.mean()) / pooled)


def print_table(rows: list[dict], columns: list[str], widths=None) -> None:
    widths = widths or {}
    head = "".join(f"{c:<{widths.get(c, 14)}}" for c in columns)
    print(head)
    print("-" * len(head))
    for r in rows:
        line = ""
        for c in columns:
            v = r.get(c, "")
            if isinstance(v, float):
                v = "nan" if not np.isfinite(v) else f"{v:.3f}"
            line += f"{str(v):<{widths.get(c, 14)}}"
        print(line)
