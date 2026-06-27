from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def corrected_rpeaks_path(qc_dir: Path, base: str) -> Path:
    return Path(qc_dir) / f"{base}_rpeaks_corrected.csv"


def save_corrected_rpeaks(path: Path, rpeaks: np.ndarray, sfreq: float) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    times = np.asarray(rpeaks, dtype=float) / float(sfreq)
    pd.DataFrame({"rpeak_time_s": times}).to_csv(path, index=False)


def load_corrected_rpeaks(path: Path, sfreq: float, n_times: int) -> np.ndarray | None:
    path = Path(path)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "rpeak_time_s" not in df.columns:
        return None
    samples = np.rint(df["rpeak_time_s"].to_numpy(dtype=float) * float(sfreq)).astype(int)
    samples = samples[(samples > 0) & (samples < int(n_times))]
    return np.unique(samples)


def selected_peak_indices(selection) -> list[int]:
    payload = _selection_payload(selection)
    points = _selection_get(payload, "points", [])
    out = set()
    for point in points or []:
        custom = point.get("customdata") if isinstance(point, dict) else None
        if custom is None and isinstance(point, dict):
            custom = point.get("curve_number", point.get("curveNumber"))
        if isinstance(custom, (list, tuple, np.ndarray)):
            custom = custom[0] if len(custom) else None
        try:
            out.add(int(custom))
        except (TypeError, ValueError):
            pass
    return sorted(out)


def selected_box_ranges(selection) -> list[tuple[float, float, float, float]]:
    boxes = _selection_get(_selection_payload(selection), "box", [])
    out = []
    for box in boxes or []:
        x = _box_get(box, "x")
        y = _box_get(box, "y")
        if x is None or y is None or len(x) < 2 or len(y) < 2:
            continue
        x0, x1 = sorted((float(x[0]), float(x[1])))
        y0, y1 = sorted((float(y[0]), float(y[1])))
        out.append((x0, x1, y0, y1))
    return out


def intersected_trace_indices(selection, x: np.ndarray, y_by_index: dict[int, np.ndarray]) -> list[int]:
    boxes = selected_box_ranges(selection)
    if not boxes:
        return []
    x = np.asarray(x, dtype=float)
    return sorted(
        idx for idx, y in y_by_index.items()
        if any(_trace_intersects_box(x, np.asarray(y, dtype=float), box) for box in boxes)
    )


def delete_excluded_rpeaks(rpeaks: np.ndarray, excluded_indices: list[int]) -> np.ndarray:
    rpeaks = np.asarray(rpeaks, dtype=int)
    if len(rpeaks) == 0:
        return rpeaks
    bad = np.asarray(sorted(set(excluded_indices)), dtype=int)
    bad = bad[(bad >= 0) & (bad < len(rpeaks))]
    if len(bad) == 0:
        return rpeaks
    keep = np.ones(len(rpeaks), dtype=bool)
    keep[bad] = False
    return rpeaks[keep]


def _selection_payload(selection):
    payload = getattr(selection, "selection", None)
    if payload is None and isinstance(selection, dict):
        payload = selection.get("selection", {})
    return payload or {}


def _selection_get(payload, key: str, default):
    if isinstance(payload, dict):
        return payload.get(key, default)
    return getattr(payload, key, default)


def _box_get(box, key: str):
    if isinstance(box, dict):
        return box.get(key)
    return getattr(box, key, None)


def _trace_intersects_box(x: np.ndarray, y: np.ndarray, box: tuple[float, float, float, float]) -> bool:
    x0, x1, y0, y1 = box
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if len(x) == 0:
        return False
    inside = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
    if bool(np.any(inside)):
        return True
    if len(x) < 2:
        return False

    xa, xb = x[:-1], x[1:]
    ya, yb = y[:-1], y[1:]
    xlo, xhi = np.minimum(xa, xb), np.maximum(xa, xb)
    ylo, yhi = np.minimum(ya, yb), np.maximum(ya, yb)
    candidates = (xhi >= x0) & (xlo <= x1) & (yhi >= y0) & (ylo <= y1)
    if not bool(np.any(candidates)):
        return False
    xa, xb, ya, yb = xa[candidates], xb[candidates], ya[candidates], yb[candidates]

    with np.errstate(divide="ignore", invalid="ignore"):
        for edge_x in (x0, x1):
            spans = (np.minimum(xa, xb) <= edge_x) & (edge_x <= np.maximum(xa, xb)) & (xa != xb)
            y_at = ya + (yb - ya) * (edge_x - xa) / (xb - xa)
            if bool(np.any(spans & (y_at >= y0) & (y_at <= y1))):
                return True
        for edge_y in (y0, y1):
            spans = (np.minimum(ya, yb) <= edge_y) & (edge_y <= np.maximum(ya, yb)) & (ya != yb)
            x_at = xa + (xb - xa) * (edge_y - ya) / (yb - ya)
            if bool(np.any(spans & (x_at >= x0) & (x_at <= x1))):
                return True
    return False


def interpolated_priors(rpeaks: np.ndarray, excluded_indices: list[int]) -> dict[int, int]:
    rpeaks = np.asarray(rpeaks, dtype=int)
    bad = np.asarray(sorted(set(excluded_indices)), dtype=int)
    bad = bad[(bad >= 0) & (bad < len(rpeaks))]
    if len(bad) == 0:
        return {}
    good_mask = np.ones(len(rpeaks), dtype=bool)
    good_mask[bad] = False
    good_idx = np.flatnonzero(good_mask)
    if len(good_idx) < 2:
        return {int(i): int(rpeaks[i]) for i in bad}
    rr = np.diff(rpeaks[good_mask])
    rr_med = int(round(np.median(rr))) if len(rr) else 0
    priors = np.interp(bad, good_idx, rpeaks[good_mask])
    left = bad < good_idx[0]
    right = bad > good_idx[-1]
    priors[left] = rpeaks[good_idx[0]] - (good_idx[0] - bad[left]) * rr_med
    priors[right] = rpeaks[good_idx[-1]] + (bad[right] - good_idx[-1]) * rr_med
    return {int(i): int(round(p)) for i, p in zip(bad, priors)}


def adaptive_refit_excluded_rpeaks(
    signal: np.ndarray,
    rpeaks: np.ndarray,
    sfreq: float,
    excluded_indices: list[int],
    search_s: float = 0.12,
) -> np.ndarray:
    """Recenter selected beats near interpolated positions using the local ECG extremum."""
    signal = np.asarray(signal, dtype=float)
    rpeaks = np.asarray(rpeaks, dtype=int)
    if len(rpeaks) == 0:
        return rpeaks
    priors = interpolated_priors(rpeaks, excluded_indices)
    if not priors:
        return rpeaks

    cleaned = signal.copy()
    finite = np.isfinite(cleaned)
    if finite.any():
        cleaned[finite] = cleaned[finite] - np.nanmedian(cleaned[finite])
    win = max(1, int(round(float(search_s) * float(sfreq))))
    out = rpeaks.copy()
    for idx, prior in priors.items():
        lo = max(0, int(prior) - win)
        hi = min(len(cleaned), int(prior) + win + 1)
        if hi <= lo:
            continue
        segment = cleaned[lo:hi]
        if not np.isfinite(segment).any():
            continue
        local = int(np.nanargmax(np.abs(segment)))
        out[idx] = lo + local
    return np.unique(out[(out > 0) & (out < len(signal))])
