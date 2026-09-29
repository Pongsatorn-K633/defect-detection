"""SubspaceAD: a PCA subspace of normal patch variation.

The whole "model" for a SKU is a mean vector and an eigenbasis. Anomaly score
is the reconstruction residual off that subspace -- what the normal parts
could not have produced.

Design rule #3 is enforced structurally: nothing here normalises a score map
by its own min and max. Per-image normalisation makes every image look like it
has its own worst defect and puts a fixed threshold out of reach. Scores are
raw residuals, optionally standardised against per-position statistics
measured on the golden set -- which is a fixed transform, not a per-image one.
"""
from __future__ import annotations

import logging

import numpy as np

from . import cancel
from .config import SubspaceConfig

log = logging.getLogger(__name__)


def fit_subspace(features: np.ndarray, tau: float = 0.97,
                 max_components: int = 512) -> dict:
    """Fit the normal subspace to stacked valid patch features.

    `features` is (N, D) -- every valid patch from every golden image. Retains
    the smallest r whose cumulative explained variance reaches `tau`.
    """
    if features.ndim != 2:
        raise ValueError(f"expected (N, D), got {features.shape}")
    n, d = features.shape
    if n < 2:
        raise ValueError(f"need at least 2 patches to fit a subspace, got {n}")

    mu = features.mean(axis=0)
    centered = features - mu

    if n > d:
        # More patches than dimensions -- every real fit: 3,000-15,000
        # patches of 1024-D. Eigendecompose the DxD covariance. The SVD of
        # the N x D data gives the same components but also builds an N x D
        # U that is thrown away, and took 60% of a fit's time -- 1.7 s per
        # call, six calls per fit with the outlier screen, on CPU. Float64
        # keeps the squared conditioning harmless for the components kept.
        cov = centered.astype(np.float64)
        cov = (cov.T @ cov) / max(n - 1, 1)
        eigenvalues, vecs = np.linalg.eigh(cov)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.clip(eigenvalues[order], 0.0, None)
        vt = vecs[:, order].T
    else:
        # Fewer patches than dimensions: the economy SVD is cheaper here and
        # better conditioned.
        _, s, vt = np.linalg.svd(centered, full_matrices=False)
        eigenvalues = (s ** 2) / max(n - 1, 1)

    total = eigenvalues.sum()
    if total <= 0:
        raise ValueError("degenerate feature set - zero variance")
    cumulative = np.cumsum(eigenvalues) / total
    r = int(np.searchsorted(cumulative, tau) + 1)
    r = max(1, min(r, max_components, len(eigenvalues)))

    return {
        "mu": mu.astype(np.float32),
        "C": vt[:r].T.astype(np.float32),          # (D, r)
        "eigenvalues": eigenvalues[:r].astype(np.float32),
        "r": r,
        "n_patches": int(n),
        "dim": int(d),
        "tau": float(tau),
        "explained": float(cumulative[r - 1]),
    }


def _flatten(features: np.ndarray):
    h, w, d = features.shape
    return features.reshape(-1, d), h, w


def residual_score(features: np.ndarray, model: dict) -> np.ndarray:
    """S(x) = ||x - (mu + C C^T (x - mu))||^2 per patch -> (h, w) map.

    The energy of the patch that the normal subspace cannot reconstruct.
    """
    x, h, w = _flatten(features)
    centered = x - model["mu"]
    coeffs = centered @ model["C"]                  # (N, r)
    recon = coeffs @ model["C"].T                   # (N, D)
    resid = centered - recon
    return np.einsum("ij,ij->i", resid, resid).reshape(h, w)


def mahalanobis_score(features: np.ndarray, model: dict) -> np.ndarray:
    """In-subspace distance: ||Lambda^-0.5 C^T (x - mu)||^2 -> (h, w) map.

    Catches defects that lie *inside* the subspace but far from the mean --
    invisible to the residual, which by construction ignores everything the
    subspace can reconstruct.
    """
    x, h, w = _flatten(features)
    centered = x - model["mu"]
    coeffs = centered @ model["C"]
    # Eigenvalues are variances along each component; floor them so a nearly
    # degenerate direction cannot blow the score up.
    lam = np.maximum(model["eigenvalues"], 1e-8)
    return ((coeffs ** 2) / lam).sum(axis=1).reshape(h, w)


def fit_position_stats(score_maps: np.ndarray, valid_masks: np.ndarray,
                       sigma_floor: float = 1e-3) -> dict:
    """Per-patch-position mean and std of the residual across golden images.

    Edges, stamped holes and formed features always score high even on perfect
    parts, because they are genuinely high-variance. Standardising each
    position against its own golden distribution drops the noise floor a long
    way and lets a single global threshold apply across the whole part.

    Positions are pooled only over the images where that patch was valid, so
    a patch masked out in one shot does not corrupt its own statistics.
    """
    maps = np.asarray(score_maps, np.float64)
    valid = np.asarray(valid_masks, bool)
    counts = valid.sum(axis=0)

    total = np.where(valid, maps, 0.0).sum(axis=0)
    mean = np.divide(total, np.maximum(counts, 1),
                     out=np.zeros_like(total), where=counts > 0)

    sq = np.where(valid, (maps - mean) ** 2, 0.0).sum(axis=0)
    var = np.divide(sq, np.maximum(counts - 1, 1),
                    out=np.zeros_like(sq), where=counts > 1)
    std = np.sqrt(var)

    # A position seen in only one image has no measurable spread; fall back to
    # the global median so it is not treated as infinitely sensitive.
    global_std = float(np.median(std[counts > 1])) if (counts > 1).any() else 1.0
    std = np.where(counts > 1, std, max(global_std, sigma_floor))
    std = np.maximum(std, max(global_std * 0.1, sigma_floor))

    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32),
            "counts": counts.astype(np.int32)}


def normalized_score(score_map: np.ndarray, position_stats: dict) -> np.ndarray:
    """z(p) = (S(p) - mu_res[p]) / sigma_res[p].

    Note this is a *fixed* transform learned at registration, not a per-image
    rescale: the same raw residual always maps to the same z, which is what
    makes a fixed threshold meaningful.
    """
    return (score_map - position_stats["mean"]) / position_stats["std"]


def aggregate(score_map: np.ndarray, valid: np.ndarray,
              method: str = "max") -> float:
    """Reduce a patch score map to one number over the valid patches.

    `max` or `top-3 mean`, not the paper's top-1% TVaR: our defects are tiny
    relative to the ROI, so averaging over the top 1% of patches dilutes a
    real defect back down into the noise. TVaR is available for comparison.
    """
    vals = score_map[valid]
    if vals.size == 0:
        return float("nan")
    if method == "max":
        return float(vals.max())
    if method == "top3mean":
        k = min(3, vals.size)
        return float(np.sort(vals)[-k:].mean())
    if method == "tvar1pct":
        k = max(1, int(round(0.01 * vals.size)))
        return float(np.sort(vals)[-k:].mean())
    raise ValueError(f"unknown aggregation {method!r}")


def score_maps(features: np.ndarray, model: dict, cfg: SubspaceConfig,
               position_stats: dict | None = None) -> dict:
    """Both score maps, standardised if position stats are available."""
    resid = residual_score(features, model)
    out = {"residual_raw": resid}
    maps = [resid if position_stats is None
            else normalized_score(resid, position_stats["residual"])]

    if cfg.use_mahalanobis:
        maha = mahalanobis_score(features, model)
        out["mahalanobis_raw"] = maha
        maps.append(maha if position_stats is None
                    else normalized_score(maha, position_stats["mahalanobis"]))

    # Final per-patch score: max(z_residual, z_mahalanobis).
    out["score"] = np.maximum.reduce(maps) if len(maps) > 1 else maps[0]
    out["normalized"] = position_stats is not None
    return out


def leave_one_out_screen(per_image_features: list[np.ndarray],
                         cfg: SubspaceConfig) -> dict:
    """Screen the golden set for contamination before saving the model.

    One defective golden part puts its defect permanently inside the normal
    subspace, and the system then cannot see that defect on any future part.
    Each candidate is scored against a subspace fitted from all the *others*;
    an outlier by robust z-score is flagged for rejection.

    Robust statistics (median / MAD) are used deliberately -- with a handful
    of golden images, one contaminated part would drag a plain mean and std
    far enough to hide itself.
    """
    k = len(per_image_features)
    if k < 3:
        return {"scores": [float("nan")] * k, "rejected": [],
                "reason": "need at least 3 golden images to screen"}

    scores = []
    for i in range(k):
        cancel.check()             # each held-out refit takes seconds
        others = np.concatenate([f for j, f in enumerate(per_image_features)
                                 if j != i], axis=0)
        try:
            model = fit_subspace(others, cfg.tau, cfg.max_components)
        except ValueError:
            scores.append(float("nan"))
            continue
        held = per_image_features[i]
        centered = held - model["mu"]
        recon = (centered @ model["C"]) @ model["C"].T
        resid = ((centered - recon) ** 2).sum(axis=1)
        # Aggregate this image the same way an ROI would be aggregated, so the
        # screen is sensitive to the same thing inspection is sensitive to.
        scores.append(float(np.sort(resid)[-max(1, len(resid) // 100):].mean()))

    arr = np.asarray(scores, float)
    finite = arr[np.isfinite(arr)]
    if len(finite) < 3:
        return {"scores": scores, "rejected": [], "reason": "too few valid scores"}

    med = float(np.median(finite))
    mad = float(np.median(np.abs(finite - med))) * 1.4826
    if mad <= 0:
        return {"scores": scores, "rejected": [], "z": [0.0] * k,
                "reason": "no spread among golden images"}

    z = (arr - med) / mad
    rejected = [i for i in range(k)
                if np.isfinite(z[i]) and z[i] > cfg.loo_reject_sigma]
    return {"scores": scores, "z": z.tolist(), "rejected": rejected,
            "median": med, "mad": mad}


def near_duplicate_pairs(per_image_features: list[np.ndarray],
                         threshold: float = 0.995) -> list[tuple[int, int, float]]:
    """Golden images that are near-identical after alignment.

    Near-duplicates cost fit time without widening the subspace usefully --
    coverage of lighting and highlight positions is what helps, not count.
    """
    means = []
    for f in per_image_features:
        m = f.mean(axis=0)
        means.append(m / (np.linalg.norm(m) + 1e-8))
    pairs = []
    for i in range(len(means)):
        for j in range(i + 1, len(means)):
            sim = float(np.dot(means[i], means[j]))
            if sim >= threshold:
                pairs.append((i, j, sim))
    return pairs
