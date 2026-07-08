#!/usr/bin/env python3
"""Statistical analysis helpers for HEPPy.

The public entry point is :func:`run_analysis_from_manifest`.  Input is a CSV
with at least participant, condition, and file_path columns.  file_path may point
to an MNE Epochs FIF or Evoked FIF.  Optional columns include group, bpm,
raw_file_path, ecg_channel, and any per-trial metadata covariates present in the
Epochs metadata.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from itertools import combinations
from pathlib import Path
import json
import math
import re
import traceback
import warnings

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy import sparse, stats
from scipy.sparse.csgraph import connected_components


@dataclass
class AnalysisConfig:
    manifest_csv: Path
    output_dir: Path
    conditions: list[str]
    condition_aliases: dict[str, list[str]] = field(default_factory=dict)
    tmin: float = 0.1
    tmax: float = 0.6
    alpha: float = 0.05
    n_permutations: int = 1024
    seed: int = 20260604
    cluster_method: str = "TFCE"
    run_omnibus: bool = True
    run_pairwise: bool = True
    run_within_groups: bool = False
    use_bpm: bool = True
    bpm_column: str = "bpm"
    min_subjects: int = 5
    montage_name: str | None = None
    montage_file: Path | None = None
    strict_channel_match: bool = False
    evoked_index: int = 0
    weight_duplicates_by_nave: bool = True
    filter_epochs_by_condition: bool = True
    epoch_metadata_condition_columns: tuple[str, ...] = (
        "condition",
        "Condition",
        "state",
        "State",
        "trial_type",
        "event",
    )
    control_mode: str = "observed"
    n_pseudotrials: int = 64
    pseudotrial_fraction: float = 0.5
    pseudotrial_tmin_from_stim_s: float = -1.1
    pseudotrial_tmax_from_stim_s: float = -0.6
    surrogate_min_shift_s: float = 0.25
    surrogate_max_shift_s: float = 0.75
    surrogate_exclusion_s: float = 0.2
    stimulus_lead_column: str | None = None
    trial_covariate_columns: list[str] = field(default_factory=list)
    tfce_start: float = 0.0
    tfce_step: float = 0.2
    tfce_h: float = 2.0
    tfce_e: float = 0.5
    rft_cluster_alpha: float = 0.05
    fig_dpi: int = 220
    save_paper_style_plots: bool = True


@dataclass(frozen=True)
class LoadedCondition:
    row_index: int
    participant: str
    group: str
    condition_original: str
    condition: str
    file_path: Path
    kind: str
    evoked: mne.Evoked
    nave: float
    n_epochs: int | None
    bpm: float | None


@dataclass(frozen=True)
class ConditionDataset:
    name: str
    participants: list[str]
    groups: list[str]
    conditions: list[str]
    X: np.ndarray
    bpm: np.ndarray | None
    times: np.ndarray
    ch_names: list[str]
    info: mne.Info


@dataclass(frozen=True)
class ClusterOutput:
    name: str
    kind: str
    method: str
    participants: list[str]
    groups: list[str]
    conditions: list[str]
    stat_obs: np.ndarray
    score_obs: np.ndarray
    p_map: np.ndarray
    sig_mask: np.ndarray
    null_max: np.ndarray
    times: np.ndarray
    ch_names: list[str]
    info: mne.Info
    adjacency: sparse.spmatrix
    components: pd.DataFrame
    extra: dict[str, object]


def safe_name(value: object) -> str:
    out = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return re.sub(r"_+", "_", out).strip("_") or "unnamed"


def normalise_label(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _aliases_for(condition: str, cfg: AnalysisConfig) -> set[str]:
    aliases = cfg.condition_aliases.get(condition, [])
    return {normalise_label(condition), *(normalise_label(a) for a in aliases)}


def canonical_condition_label(value: object, cfg: AnalysisConfig) -> str:
    norm = normalise_label(value)
    matches = [c for c in cfg.conditions if norm in _aliases_for(c, cfg)]
    if len(matches) == 1:
        return matches[0]
    if not matches and str(value) in cfg.conditions:
        return str(value)
    if len(matches) > 1:
        raise ValueError(f"Condition {value!r} maps ambiguously to {matches}.")
    raise ValueError(f"Condition {value!r} does not match any configured condition.")


def _resolve_path(path_value: object, csv_dir: Path) -> Path:
    p = Path(str(path_value).strip().strip('"').strip("'")).expanduser()
    return p if p.is_absolute() else csv_dir / p


def _read_manifest(cfg: AnalysisConfig) -> pd.DataFrame:
    df = pd.read_csv(cfg.manifest_csv)
    required = {"participant", "condition", "file_path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Analysis manifest is missing required columns: {sorted(missing)}")
    df = df.copy()
    if "group" not in df.columns:
        df["group"] = "all"
    df["participant"] = df["participant"].astype(str).str.strip()
    df["group"] = df["group"].astype(str).str.strip()
    df["condition_original"] = df["condition"].astype(str).str.strip()
    df["condition"] = df["condition_original"].map(lambda x: canonical_condition_label(x, cfg))
    df["file_path"] = df["file_path"].map(lambda x: _resolve_path(x, cfg.manifest_csv.parent))
    for col in ("raw_file_path",):
        if col in df.columns:
            df[col] = df[col].map(lambda x: "" if pd.isna(x) else str(_resolve_path(x, cfg.manifest_csv.parent)))
    if cfg.bpm_column in df.columns:
        df[cfg.bpm_column] = pd.to_numeric(df[cfg.bpm_column], errors="coerce")
    missing_files = [str(p) for p in df["file_path"] if not Path(p).exists()]
    if missing_files:
        raise FileNotFoundError("Missing analysis file(s):\n" + "\n".join(missing_files[:20]))
    return df


def _apply_montage(obj, cfg: AnalysisConfig):
    if cfg.montage_file is None and cfg.montage_name in (None, "", "None"):
        return obj
    out = obj.copy()
    montage = (
        mne.channels.read_custom_montage(cfg.montage_file)
        if cfg.montage_file
        else mne.channels.make_standard_montage(str(cfg.montage_name))
    )
    out.set_montage(montage, on_missing="warn")
    return out


def _pick_eeg(obj):
    out = obj.copy()
    picks = mne.pick_types(out.info, eeg=True, meg=False, eog=False, ecg=False, stim=False, misc=False, exclude=[])
    if len(picks) == 0:
        raise RuntimeError("No EEG channels found.")
    out.pick(picks)
    return out


def _select_epochs_by_condition(epochs: mne.Epochs, row_condition: str, condition: str, cfg: AnalysisConfig) -> mne.Epochs:
    if not cfg.filter_epochs_by_condition:
        return epochs
    aliases = _aliases_for(condition, cfg) | {normalise_label(row_condition)}
    if epochs.metadata is not None and len(epochs.metadata) == len(epochs):
        for col in cfg.epoch_metadata_condition_columns:
            if col in epochs.metadata.columns:
                mask = epochs.metadata[col].map(normalise_label).isin(aliases).to_numpy()
                if mask.any():
                    return epochs[np.flatnonzero(mask)]
    matches = []
    for event_name in epochs.event_id:
        norm = normalise_label(event_name)
        if norm in aliases or any(a and (a in norm or norm in a) for a in aliases):
            matches.append(event_name)
    if matches:
        try:
            return epochs[sorted(set(matches))]
        except Exception:
            pass
    warnings.warn(
        f"Could not filter epochs by condition {row_condition!r}; averaging all epochs in {epochs.filename}.",
        RuntimeWarning,
    )
    return epochs


def _residualise_epochs_by_metadata(epochs: mne.Epochs, covariates: list[str]) -> mne.Epochs:
    if not covariates:
        return epochs
    if epochs.metadata is None:
        raise ValueError("Per-trial covariate correction was requested, but Epochs metadata is missing.")
    use_cols = [c for c in covariates if c and c in epochs.metadata.columns]
    if not use_cols:
        raise ValueError(f"None of the requested per-trial covariates exist in metadata: {covariates}")
    meta = epochs.metadata[use_cols].apply(pd.to_numeric, errors="coerce")
    ok = np.isfinite(meta.to_numpy()).all(axis=1)
    if ok.sum() < 3:
        raise ValueError("Too few valid trials for per-trial covariate correction.")
    ep = epochs[np.flatnonzero(ok)].copy()
    Z = meta.loc[ok].to_numpy(dtype=float)
    Z = (Z - Z.mean(axis=0, keepdims=True)) / np.maximum(Z.std(axis=0, ddof=1, keepdims=True), 1e-12)
    X = np.column_stack([np.ones(len(Z)), Z])
    data = ep.get_data(copy=True)
    flat = data.reshape(data.shape[0], -1)
    beta = np.linalg.pinv(X) @ flat
    fitted_cov = X[:, 1:] @ beta[1:]
    corrected = (flat - fitted_cov).reshape(data.shape)
    ep._data = corrected
    return ep


def _auto_pick_ecg(raw: mne.io.BaseRaw, preferred: str | None) -> str:
    if preferred and preferred in raw.ch_names:
        return preferred
    idx = mne.pick_types(raw.info, ecg=True)
    if len(idx):
        return raw.ch_names[idx[0]]
    for ch in raw.ch_names:
        if any(tok in ch.upper() for tok in ("ECG", "EKG", "CARD")):
            return ch
    raise RuntimeError("No ECG channel found for surrogate heartbeat control.")


def _canonical_raw_channel(ch_name: str) -> str:
    name = re.sub(r"^EEG\s+", "", ch_name.strip(), flags=re.I)
    name = name.split("-")[0].strip()
    replacements = {"FP1": "Fp1", "FP2": "Fp2", "FZ": "Fz", "CZ": "Cz", "PZ": "Pz", "OZ": "Oz"}
    return replacements.get(name.upper(), name)


def _prepare_raw_for_surrogates(raw: mne.io.BaseRaw, cfg: AnalysisConfig) -> mne.io.BaseRaw:
    raw = raw.copy()
    raw.rename_channels({ch: _canonical_raw_channel(ch) for ch in raw.ch_names})
    type_map = {}
    for ch in raw.ch_names:
        up = ch.upper()
        if "EKG" in up or "ECG" in up:
            type_map[ch] = "ecg"
        elif "PHOTIC" in up or up == "PH":
            type_map[ch] = "stim"
        elif up.startswith("PG") or up in {"SP1", "SP2", "RLC", "LUC", "T1", "T2"} or up.isdigit():
            type_map[ch] = "misc"
        else:
            type_map[ch] = "eeg"
    raw.set_channel_types(type_map, verbose="ERROR")
    if cfg.montage_file is not None:
        raw.set_montage(mne.channels.read_custom_montage(cfg.montage_file), on_missing="ignore", verbose="ERROR")
    elif cfg.montage_name not in (None, "", "None"):
        raw.set_montage(mne.channels.make_standard_montage(str(cfg.montage_name)), on_missing="ignore", verbose="ERROR")
    else:
        raw.set_montage(mne.channels.make_standard_montage("standard_1020"), on_missing="ignore", verbose="ERROR")
    return raw


def _read_prepared_raw(row: pd.Series, cfg: AnalysisConfig) -> mne.io.BaseRaw:
    raw_path = row.get("raw_file_path", "")
    if not raw_path:
        raise ValueError("This control requires a raw_file_path column.")
    suffix = Path(str(raw_path)).suffix.lower()
    if suffix == ".fif":
        raw = mne.io.read_raw_fif(str(raw_path), preload=True, verbose="ERROR")
    elif suffix == ".edf":
        raw = mne.io.read_raw_edf(str(raw_path), preload=True, verbose="ERROR")
    elif suffix == ".bdf":
        raw = mne.io.read_raw_bdf(str(raw_path), preload=True, verbose="ERROR")
    elif suffix == ".vhdr":
        raw = mne.io.read_raw_brainvision(str(raw_path), preload=True, verbose="ERROR")
    else:
        raise ValueError(f"Unsupported raw_file_path type: {raw_path}")
    return _prepare_raw_for_surrogates(raw, cfg)


def _stim_and_rpeak_samples_from_epochs(epochs: mne.Epochs, sf: float) -> tuple[np.ndarray | None, np.ndarray]:
    rpeak_samples = epochs.events[:, 0].astype(int)
    stim_samples = None
    meta = epochs.metadata
    if meta is not None:
        for stim_col in ("stimulus_sample", "stim_sample", "target_sample"):
            if stim_col in meta.columns:
                stim_samples = pd.to_numeric(meta[stim_col], errors="coerce").to_numpy(dtype=float)
                break
        if stim_samples is None:
            for stim_col in ("stimulus_time_s", "stim_time_s", "target_time_s"):
                if stim_col in meta.columns:
                    stim_samples = pd.to_numeric(meta[stim_col], errors="coerce").to_numpy(dtype=float) * sf
                    break
        for r_col in ("rpeak_sample", "heartbeat_sample"):
            if r_col in meta.columns:
                rpeak_samples = pd.to_numeric(meta[r_col], errors="coerce").to_numpy(dtype=float).astype(int)
                break
        if stim_samples is None:
            for lead_col in ("stimulus_lead_s", "stimulus_lead", "lead_s"):
                if lead_col in meta.columns:
                    lead = pd.to_numeric(meta[lead_col], errors="coerce").to_numpy(dtype=float)
                    stim_samples = rpeak_samples + lead * sf
                    break
    if stim_samples is not None:
        ok = np.isfinite(stim_samples) & np.isfinite(rpeak_samples)
        stim_samples = stim_samples[ok].astype(int)
        rpeak_samples = rpeak_samples[ok].astype(int)
    return stim_samples, rpeak_samples


def _epoch_raw_at_samples(raw: mne.io.BaseRaw, samples: np.ndarray, template: mne.Epochs, cfg: AnalysisConfig, label: str) -> mne.Epochs:
    sf = float(raw.info["sfreq"])
    epoch_tmin = float(template.tmin)
    epoch_tmax = float(template.tmax)
    lo = int(abs(epoch_tmin) * sf) + 1
    hi = raw.n_times - int(abs(epoch_tmax) * sf) - 1
    samples = np.unique(samples[(samples > lo) & (samples < hi)].astype(int))
    if len(samples) == 0:
        raise RuntimeError(f"No valid {label} events remain inside the raw recording.")
    events = np.column_stack([samples, np.zeros(len(samples), dtype=int), np.ones(len(samples), dtype=int)])
    epochs = mne.Epochs(raw, events, event_id={label: 1}, tmin=epoch_tmin, tmax=epoch_tmax, baseline=None, preload=True, picks="eeg", verbose="ERROR")
    common = [ch for ch in template.ch_names if ch in epochs.ch_names]
    epochs.pick(common).reorder_channels(common)
    return epochs


def _pseudotrial_corrected_evoked(row: pd.Series, epochs: mne.Epochs, cfg: AnalysisConfig, seed_offset: int) -> mne.Evoked:
    raw = _read_prepared_raw(row, cfg)
    sf = float(raw.info["sfreq"])
    stim_samples, _ = _stim_and_rpeak_samples_from_epochs(epochs, sf)
    if stim_samples is None or len(stim_samples) == 0:
        raise ValueError(
            "Pseudotrial correction requires stimulus timing metadata: "
            "stimulus_sample/stimulus_time_s or stimulus_lead_s."
        )
    rng = np.random.default_rng(cfg.seed + seed_offset)
    pseudo_means = []
    for _ in range(max(1, cfg.n_pseudotrials)):
        offsets = rng.uniform(abs(cfg.pseudotrial_tmax_from_stim_s), abs(cfg.pseudotrial_tmin_from_stim_s), size=len(stim_samples))
        pseudo_samples = np.round(stim_samples - offsets * sf).astype(int)
        pseudo_epochs = _epoch_raw_at_samples(raw, pseudo_samples, epochs, cfg, "pseudotrial")
        pseudo_means.append(pseudo_epochs.average().data)
    real = epochs.average()
    pseudo_mean = np.mean(pseudo_means, axis=0)
    real.data = real.data - pseudo_mean
    real.comment = f"pseudotrial_corrected:window={cfg.pseudotrial_tmin_from_stim_s}:{cfg.pseudotrial_tmax_from_stim_s}:reps={cfg.n_pseudotrials}"
    return real


def _surrogate_epochs_from_raw(row: pd.Series, template: mne.Epochs, cfg: AnalysisConfig, seed_offset: int) -> mne.Epochs:
    raw = _read_prepared_raw(row, cfg)
    sf = float(raw.info["sfreq"])
    stim_samples, rpeak_samples = _stim_and_rpeak_samples_from_epochs(template, sf)
    if stim_samples is not None and len(stim_samples) > 1:
        latencies = rpeak_samples - stim_samples
        rng = np.random.default_rng(cfg.seed + seed_offset)
        surrogate_samples = stim_samples + rng.permutation(latencies)
        return _epoch_raw_at_samples(raw, surrogate_samples.astype(int), template, cfg, "surrogate")

    ecg_name = _auto_pick_ecg(raw, str(row.get("ecg_channel", "") or "") or None)
    try:
        import neurokit2 as nk

        sig = raw.get_data(picks=ecg_name)[0]
        rpeaks = np.asarray(nk.ecg_process(sig, sampling_rate=sf)[1]["ECG_R_Peaks"], dtype=int)
    except Exception as exc:
        raise RuntimeError(f"Could not detect ECG peaks for surrogate control: {exc}") from exc
    rng = np.random.default_rng(cfg.seed + seed_offset)
    shifts = rng.uniform(cfg.surrogate_min_shift_s, cfg.surrogate_max_shift_s, size=len(rpeaks))
    signs = rng.choice(np.array([-1.0, 1.0]), size=len(rpeaks))
    sur = np.round(rpeaks + signs * shifts * sf).astype(int)
    lo = int(abs(cfg.tmin) * sf) + 1
    hi = raw.n_times - int(abs(cfg.tmax) * sf) - 1
    sur = sur[(sur > lo) & (sur < hi)]
    if len(sur) == 0:
        raise RuntimeError("No surrogate events remain inside the recording.")
    true = rpeaks[(rpeaks > 0) & (rpeaks < raw.n_times)]
    keep = np.ones(len(sur), dtype=bool)
    exclusion = int(round(cfg.surrogate_exclusion_s * sf))
    for i, sample in enumerate(sur):
        if np.min(np.abs(true - sample)) <= exclusion:
            keep[i] = False
    sur = np.unique(sur[keep])
    events = np.column_stack([sur, np.zeros(len(sur), dtype=int), np.ones(len(sur), dtype=int)])
    epochs = mne.Epochs(raw, events, event_id={"surrogate": 1}, tmin=cfg.tmin, tmax=cfg.tmax, baseline=None, preload=True, picks="eeg", verbose="ERROR")
    epochs.pick(template.ch_names)
    return epochs


def _read_file_as_evoked(row: pd.Series, cfg: AnalysisConfig) -> tuple[mne.Evoked, str, float, int | None]:
    path = Path(row["file_path"])
    condition = str(row["condition"])
    original = str(row["condition_original"])
    covariates = list(cfg.trial_covariate_columns)
    if cfg.stimulus_lead_column:
        covariates.append(cfg.stimulus_lead_column)

    errors = []
    try:
        epochs = mne.read_epochs(path, preload=True, verbose="ERROR")
        epochs = _pick_eeg(epochs)
        epochs = _apply_montage(epochs, cfg)
        epochs = _select_epochs_by_condition(epochs, original, condition, cfg)
        if len(epochs) == 0:
            raise RuntimeError("Condition selection produced zero epochs.")
        if cfg.control_mode == "surrogate_heartbeat":
            epochs = _surrogate_epochs_from_raw(row, epochs, cfg, int(row.name))
        if covariates:
            epochs = _residualise_epochs_by_metadata(epochs, covariates)
        if cfg.control_mode == "pseudotrial":
            evoked = _pseudotrial_corrected_evoked(row, epochs, cfg, int(row.name))
        else:
            evoked = epochs.average()
        evoked = _pick_eeg(_apply_montage(evoked, cfg)).crop(cfg.tmin, cfg.tmax, include_tmax=True)
        evoked.comment = f"{cfg.control_mode}:{condition}"
        return evoked, "epochs", float(max(evoked.nave, 1)), int(len(epochs))
    except Exception as exc:
        errors.append(f"Epochs failed: {type(exc).__name__}: {exc}")
    try:
        evokeds = mne.read_evokeds(path, condition=None, baseline=None, verbose="ERROR")
        if cfg.evoked_index >= len(evokeds):
            raise RuntimeError(f"evoked_index={cfg.evoked_index}, file has {len(evokeds)} evokeds.")
        evoked = _pick_eeg(_apply_montage(evokeds[cfg.evoked_index], cfg)).crop(cfg.tmin, cfg.tmax, include_tmax=True)
        if cfg.control_mode in ("pseudotrial", "surrogate_heartbeat") or covariates:
            warnings.warn(f"{cfg.control_mode} / per-trial correction needs Epochs input; using Evoked as-is for {path.name}.")
        return evoked, "evoked", float(max(evoked.nave, 1)), None
    except Exception as exc:
        errors.append(f"Evoked failed: {type(exc).__name__}: {exc}")
    raise RuntimeError(f"Could not read {path}. " + " | ".join(errors))


def _load_rows(df: pd.DataFrame, cfg: AnalysisConfig) -> list[LoadedCondition]:
    loaded = []
    failures = []
    for row_index, row in df.iterrows():
        try:
            evoked, kind, nave, n_epochs = _read_file_as_evoked(row, cfg)
            bpm = None
            if cfg.use_bpm and cfg.bpm_column in df.columns and not pd.isna(row[cfg.bpm_column]):
                bpm = float(row[cfg.bpm_column])
            loaded.append(
                LoadedCondition(
                    row_index=int(row_index),
                    participant=str(row["participant"]),
                    group=str(row["group"]),
                    condition_original=str(row["condition_original"]),
                    condition=str(row["condition"]),
                    file_path=Path(row["file_path"]),
                    kind=kind,
                    evoked=evoked,
                    nave=nave,
                    n_epochs=n_epochs,
                    bpm=bpm,
                )
            )
        except Exception as exc:
            failures.append(
                {
                    "row_index": int(row_index),
                    "participant": row.get("participant", ""),
                    "condition": row.get("condition", ""),
                    "file_path": str(row.get("file_path", "")),
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
            )
    if failures:
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(failures).to_csv(cfg.output_dir / "load_failures.csv", index=False)
        raise RuntimeError(f"Failed to load {len(failures)} manifest row(s). See load_failures.csv.")
    return loaded


def _weighted_evoked_mean(evokeds: list[mne.Evoked], cfg: AnalysisConfig, comment: str) -> mne.Evoked:
    template = evokeds[0].copy()
    weights = np.array([max(float(ev.nave), 1.0) for ev in evokeds], dtype=float)
    if not cfg.weight_duplicates_by_nave:
        weights = np.ones_like(weights)
    data = sum(ev.data * w for ev, w in zip(evokeds, weights)) / weights.sum()
    template.data = data
    template.nave = int(round(weights.sum()))
    template.comment = comment
    return template


def _check_positions(info: mne.Info) -> None:
    bad = []
    for ch in info["chs"]:
        if ch["kind"] == mne.io.constants.FIFF.FIFFV_EEG_CH:
            loc = ch["loc"][:3]
            if not np.all(np.isfinite(loc)) or np.allclose(loc, 0):
                bad.append(ch["ch_name"])
    if bad:
        raise RuntimeError(f"Missing EEG channel locations; set montage_name/file. Examples: {bad[:10]}")


def _harmonise_evokeds(evokeds: list[mne.Evoked], cfg: AnalysisConfig) -> tuple[list[mne.Evoked], list[str], np.ndarray, mne.Info]:
    ref = evokeds[0]
    times = ref.times.copy()
    if cfg.strict_channel_match:
        ch_names = ref.ch_names
    else:
        common = set(ref.ch_names)
        for ev in evokeds[1:]:
            common &= set(ev.ch_names)
        ch_names = [ch for ch in ref.ch_names if ch in common]
    if len(ch_names) < 2:
        raise RuntimeError("Fewer than two common EEG channels across files.")
    out = []
    for ev in evokeds:
        ev2 = ev.copy().pick(ch_names).reorder_channels(ch_names)
        if not np.allclose(ev2.times, times, rtol=0, atol=1e-12):
            raise RuntimeError("Time vectors differ across files after cropping.")
        out.append(ev2)
    info = out[0].info.copy()
    _check_positions(info)
    return out, ch_names, times, info


def _build_dataset(loaded: list[LoadedCondition], cfg: AnalysisConfig, name: str, group_filter: str | None = None) -> ConditionDataset:
    rows = [r for r in loaded if group_filter is None or r.group == group_filter]
    by_pc: dict[tuple[str, str], list[LoadedCondition]] = {}
    groups_by_p: dict[str, set[str]] = {}
    bpm_by_pc: dict[tuple[str, str], list[float]] = {}
    for r in rows:
        by_pc.setdefault((r.participant, r.condition), []).append(r)
        groups_by_p.setdefault(r.participant, set()).add(r.group)
        if r.bpm is not None:
            bpm_by_pc.setdefault((r.participant, r.condition), []).append(r.bpm)
    participants = []
    skipped = []
    for p in sorted(groups_by_p):
        missing = [c for c in cfg.conditions if (p, c) not in by_pc]
        if missing:
            skipped.append({"participant": p, "reason": "missing_conditions", "missing": ";".join(missing)})
            continue
        if len(groups_by_p[p]) != 1:
            skipped.append({"participant": p, "reason": "multiple_groups"})
            continue
        participants.append(p)
    if skipped:
        pd.DataFrame(skipped).to_csv(cfg.output_dir / f"skipped_participants_{safe_name(name)}.csv", index=False)
    if len(participants) < cfg.min_subjects:
        raise RuntimeError(f"{name}: only {len(participants)} complete participants; minimum is {cfg.min_subjects}.")
    all_evokeds = []
    cell_evoked = {}
    for p in participants:
        for c in cfg.conditions:
            ev = _weighted_evoked_mean([r.evoked for r in by_pc[(p, c)]], cfg, f"{p}:{c}")
            cell_evoked[(p, c)] = ev
            all_evokeds.append(ev)
    harmonised, ch_names, times, info = _harmonise_evokeds(all_evokeds, cfg)
    h_iter = iter(harmonised)
    for p in participants:
        for c in cfg.conditions:
            cell_evoked[(p, c)] = next(h_iter)
    X = np.empty((len(participants), len(cfg.conditions), len(times), len(ch_names)), dtype=float)
    groups = []
    bpm = None
    use_bpm = cfg.use_bpm and any(r.bpm is not None for r in rows)
    if use_bpm:
        bpm = np.full((len(participants), len(cfg.conditions)), np.nan)
    for si, p in enumerate(participants):
        groups.append(sorted(groups_by_p[p])[0])
        for ci, c in enumerate(cfg.conditions):
            X[si, ci] = cell_evoked[(p, c)].data.T
            if bpm is not None and (p, c) in bpm_by_pc:
                bpm[si, ci] = float(np.mean(bpm_by_pc[(p, c)]))
    if bpm is not None and not np.all(np.isfinite(bpm)):
        warnings.warn("BPM was requested but incomplete; disabling BPM correction for this dataset.")
        bpm = None
    return ConditionDataset(name, participants, groups, list(cfg.conditions), X, bpm, times, ch_names, info)


def _condition_data_to_Y(dataset: ConditionDataset):
    n_subj, n_cond, n_times, n_ch = dataset.X.shape
    Y = dataset.X.reshape(n_subj * n_cond, n_times * n_ch)
    subject_idx = np.repeat(np.arange(n_subj), n_cond)
    condition_idx = np.tile(np.arange(n_cond), n_subj)
    bpm_vec = None if dataset.bpm is None else dataset.bpm.reshape(n_subj * n_cond)
    return Y, subject_idx, condition_idx, bpm_vec


def _subject_design(subject_idx: np.ndarray, n_subjects: int) -> np.ndarray:
    Z = np.zeros((len(subject_idx), n_subjects))
    Z[np.arange(len(subject_idx)), subject_idx] = 1.0
    return Z


def _condition_design(condition_idx: np.ndarray, n_conditions: int) -> np.ndarray:
    C = np.zeros((len(condition_idx), n_conditions - 1))
    for k in range(1, n_conditions):
        C[:, k - 1] = condition_idx == k
    return C


def _bpm_design(bpm_vec: np.ndarray | None) -> np.ndarray | None:
    if bpm_vec is None:
        return None
    sd = float(np.std(bpm_vec, ddof=1))
    if sd <= 0 or not np.isfinite(sd):
        return None
    return ((bpm_vec - float(np.mean(bpm_vec))) / sd)[:, None]


def _fit_values(design: np.ndarray, Y: np.ndarray):
    beta = np.linalg.pinv(design) @ Y
    fitted = design @ beta
    return fitted, Y - fitted, int(np.linalg.matrix_rank(design))


def _glm_nested_f(Y: np.ndarray, full: np.ndarray, reduced: np.ndarray) -> np.ndarray:
    _, rf, rank_f = _fit_values(full, Y)
    _, rr, rank_r = _fit_values(reduced, Y)
    rss_f = np.sum(rf * rf, axis=0)
    rss_r = np.sum(rr * rr, axis=0)
    df1 = rank_f - rank_r
    df2 = Y.shape[0] - rank_f
    if df1 <= 0 or df2 <= 0:
        raise RuntimeError(f"Invalid GLM degrees of freedom: df1={df1}, df2={df2}")
    F = ((np.maximum(rss_r - rss_f, 0.0) / df1) / np.maximum(rss_f / df2, np.finfo(float).tiny))
    return np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)


def _paired_t(Y: np.ndarray, bpm_diff: np.ndarray | None = None) -> np.ndarray:
    if bpm_diff is None:
        t = stats.ttest_1samp(Y, 0.0, axis=0, nan_policy="omit").statistic
        return np.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
    sd = float(np.std(bpm_diff, ddof=1))
    if sd <= 0 or not np.isfinite(sd):
        return _paired_t(Y, None)
    bpm_z = ((bpm_diff - float(np.mean(bpm_diff))) / sd)[:, None]
    X = np.column_stack([np.ones(len(bpm_z)), bpm_z])
    beta = np.linalg.pinv(X) @ Y
    resid = Y - X @ beta
    df = Y.shape[0] - int(np.linalg.matrix_rank(X))
    rss = np.sum(resid * resid, axis=0)
    cvar = float(np.linalg.pinv(X.T @ X)[0, 0])
    t = beta[0] / np.maximum(np.sqrt(np.maximum(rss / df, 0.0) * cvar), np.finfo(float).tiny)
    return np.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)


def _sensor_adjacency(info: mne.Info, ch_names: list[str]) -> sparse.spmatrix:
    try:
        adj, adj_names = mne.channels.find_ch_adjacency(info, ch_type="eeg")
        if list(adj_names) != ch_names:
            idx = [list(adj_names).index(ch) for ch in ch_names]
            adj = adj[idx][:, idx]
        return adj.astype(bool).tocsr()
    except ValueError:
        n = len(ch_names)
        return sparse.diags([np.ones(n - 1), np.ones(n - 1)], [-1, 1], shape=(n, n), format="csr", dtype=bool)


def _time_channel_adjacency(sensor_adj: sparse.spmatrix, n_times: int) -> sparse.spmatrix:
    time_adj = sparse.diags([np.ones(n_times - 1), np.ones(n_times - 1)], [-1, 1], shape=(n_times, n_times), format="csr", dtype=bool)
    full = sparse.kron(time_adj, sparse.eye(sensor_adj.shape[0], dtype=bool), format="csr")
    full = full + sparse.kron(sparse.eye(n_times, dtype=bool), sensor_adj.astype(bool), format="csr")
    full.setdiag(False)
    full.eliminate_zeros()
    return full.astype(bool).tocsr()


def _tfce_one_sided(values: np.ndarray, adjacency: sparse.spmatrix, cfg: AnalysisConfig) -> np.ndarray:
    vals = np.maximum(np.nan_to_num(values.astype(float), nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    out = np.zeros_like(vals)
    max_val = float(np.max(vals))
    if max_val <= cfg.tfce_start:
        return out
    for h in np.arange(cfg.tfce_start + cfg.tfce_step, max_val + cfg.tfce_step * 0.5, cfg.tfce_step):
        active_idx = np.flatnonzero(vals >= h)
        if len(active_idx) == 0:
            continue
        n_comp, labels = connected_components(adjacency[active_idx][:, active_idx], directed=False)
        extents = np.bincount(labels).astype(float)
        out[active_idx] += (extents[labels] ** cfg.tfce_e) * (h ** cfg.tfce_h) * cfg.tfce_step
    return out


def _tfce(stat_map: np.ndarray, adjacency: sparse.spmatrix, two_tailed: bool, cfg: AnalysisConfig) -> np.ndarray:
    flat = stat_map.reshape(-1)
    if two_tailed:
        return (_tfce_one_sided(flat, adjacency, cfg) - _tfce_one_sided(-flat, adjacency, cfg)).reshape(stat_map.shape)
    return _tfce_one_sided(flat, adjacency, cfg).reshape(stat_map.shape)


def _cluster_score(stat_map: np.ndarray, adjacency: sparse.spmatrix, two_tailed: bool, cfg: AnalysisConfig, df: tuple[int, int] | int) -> np.ndarray:
    flat = stat_map.reshape(-1)
    if two_tailed:
        threshold = stats.t.ppf(1.0 - cfg.rft_cluster_alpha / 2.0, int(df))
        active = np.abs(flat) >= threshold
        values = np.abs(flat)
    else:
        df1, df2 = df
        threshold = stats.f.ppf(1.0 - cfg.rft_cluster_alpha, int(df1), int(df2))
        active = flat >= threshold
        values = flat
    out = np.zeros_like(flat, dtype=float)
    active_idx = np.flatnonzero(active)
    if len(active_idx):
        n_comp, labels = connected_components(adjacency[active_idx][:, active_idx], directed=False)
        for comp in range(n_comp):
            idx = active_idx[labels == comp]
            out[idx] = float(np.sum(values[idx]))
    if two_tailed:
        out *= np.sign(flat)
    return out.reshape(stat_map.shape)


def _score(stat_map: np.ndarray, adjacency: sparse.spmatrix, two_tailed: bool, cfg: AnalysisConfig, df) -> np.ndarray:
    if cfg.cluster_method.upper() == "TFCE":
        return _tfce(stat_map, adjacency, two_tailed, cfg)
    return _cluster_score(stat_map, adjacency, two_tailed, cfg, df)


def _score_label(cfg: AnalysisConfig) -> str:
    method = cfg.cluster_method.upper()
    if method == "TFCE":
        return "TFCE score"
    if method in ("CLUSTER_SUM", "CBP", "PAPER_CBP"):
        return "cluster sum statistic"
    return "cluster statistic"


def _p_map(score_obs: np.ndarray, null_max: np.ndarray, two_tailed: bool) -> np.ndarray:
    obs = np.abs(score_obs) if two_tailed else score_obs
    flat = obs.reshape(-1)
    p = np.empty_like(flat, dtype=float)
    for i, value in enumerate(flat):
        p[i] = (1.0 + float(np.sum(null_max >= value))) / (len(null_max) + 1.0)
    return p.reshape(obs.shape)


def _summarise_components(sig_mask, stat_map, p_map, times, ch_names, adjacency, stat_label):
    cols = ["component", "n_points", "t_start", "t_end", "n_channels", "channels", "peak_time", "peak_channel", f"peak_{stat_label}", "min_p", f"mean_{stat_label}"]
    if not sig_mask.any():
        return pd.DataFrame(columns=cols)
    sig_idx = np.flatnonzero(sig_mask.reshape(-1))
    n_comp, labels = connected_components(adjacency[sig_idx][:, sig_idx], directed=False)
    rows = []
    for comp in range(n_comp):
        flat_idx = sig_idx[labels == comp]
        ti, ci = np.unravel_index(flat_idx, sig_mask.shape)
        vals = stat_map[ti, ci]
        peak = int(np.argmax(np.abs(vals)))
        chans = [ch_names[i] for i in sorted(set(ci.tolist()))]
        rows.append({
            "component": comp + 1,
            "n_points": int(len(flat_idx)),
            "t_start": float(times[ti.min()]),
            "t_end": float(times[ti.max()]),
            "n_channels": int(len(chans)),
            "channels": ";".join(chans),
            "peak_time": float(times[ti[peak]]),
            "peak_channel": ch_names[int(ci[peak])],
            f"peak_{stat_label}": float(vals[peak]),
            "min_p": float(np.min(p_map[ti, ci])),
            f"mean_{stat_label}": float(np.mean(vals)),
        })
    return pd.DataFrame(rows, columns=cols).sort_values(["min_p", "n_points"], ascending=[True, False]).reset_index(drop=True)


def _blocked_perm_indices(subject_idx: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.arange(len(subject_idx))
    for s in np.unique(subject_idx):
        rows = np.flatnonzero(subject_idx == s)
        out[rows] = rng.permutation(rows)
    return out


def _run_omnibus(dataset: ConditionDataset, cfg: AnalysisConfig, name: str) -> ClusterOutput:
    sensor_adj = _sensor_adjacency(dataset.info, dataset.ch_names)
    adj = _time_channel_adjacency(sensor_adj, len(dataset.times))
    Y, subject_idx, condition_idx, bpm_vec = _condition_data_to_Y(dataset)
    Zs = _subject_design(subject_idx, len(dataset.participants))
    C = _condition_design(condition_idx, len(dataset.conditions))
    Zb = _bpm_design(bpm_vec)
    reduced = Zs if Zb is None else np.column_stack([Zs, Zb])
    full = np.column_stack([reduced, C])
    stat = _glm_nested_f(Y, full, reduced).reshape(len(dataset.times), len(dataset.ch_names))
    df = (int(np.linalg.matrix_rank(full) - np.linalg.matrix_rank(reduced)), int(Y.shape[0] - np.linalg.matrix_rank(full)))
    score_obs = _score(stat, adj, False, cfg, df)
    fitted_red, resid_red, _ = _fit_values(reduced, Y)
    rng = np.random.default_rng(cfg.seed)
    null_max = np.empty(cfg.n_permutations)
    for pi in range(cfg.n_permutations):
        Yp = fitted_red + resid_red[_blocked_perm_indices(subject_idx, rng)]
        sp = _glm_nested_f(Yp, full, reduced).reshape(stat.shape)
        null_max[pi] = float(np.max(_score(sp, adj, False, cfg, df)))
    p_map = _p_map(score_obs, null_max, False)
    sig = p_map < cfg.alpha
    return ClusterOutput(name, "omnibus_f", cfg.cluster_method.upper(), dataset.participants, sorted(set(dataset.groups)), dataset.conditions, stat, score_obs, p_map, sig, null_max, dataset.times, dataset.ch_names, dataset.info, adj, _summarise_components(sig, stat, p_map, dataset.times, dataset.ch_names, adj, "F"), {"uses_bpm": Zb is not None, "df": df})


def _run_pairwise(dataset: ConditionDataset, cfg: AnalysisConfig, cond_a: str, cond_b: str, name: str) -> ClusterOutput:
    ia, ib = dataset.conditions.index(cond_a), dataset.conditions.index(cond_b)
    sensor_adj = _sensor_adjacency(dataset.info, dataset.ch_names)
    adj = _time_channel_adjacency(sensor_adj, len(dataset.times))
    D = dataset.X[:, ia] - dataset.X[:, ib]
    Y = D.reshape(D.shape[0], -1)
    bpm_diff = None if dataset.bpm is None else dataset.bpm[:, ia] - dataset.bpm[:, ib]
    stat = _paired_t(Y, bpm_diff).reshape(len(dataset.times), len(dataset.ch_names))
    df = Y.shape[0] - (2 if bpm_diff is not None and np.std(bpm_diff, ddof=1) > 0 else 1)
    score_obs = _score(stat, adj, True, cfg, df)
    rng = np.random.default_rng(cfg.seed + 17 + ia * 101 + ib)
    null_max = np.empty(cfg.n_permutations)
    if bpm_diff is None or np.std(bpm_diff, ddof=1) <= 0:
        fitted_red = np.zeros_like(Y)
        resid_red = Y
        bpm_for_stat = None
    else:
        z = ((bpm_diff - bpm_diff.mean()) / bpm_diff.std(ddof=1))[:, None]
        fitted_red, resid_red, _ = _fit_values(z, Y)
        bpm_for_stat = bpm_diff
    for pi in range(cfg.n_permutations):
        signs = rng.choice(np.array([-1.0, 1.0]), size=Y.shape[0])[:, None]
        tp = _paired_t(fitted_red + resid_red * signs, bpm_for_stat).reshape(stat.shape)
        null_max[pi] = float(np.max(np.abs(_score(tp, adj, True, cfg, df))))
    p_map = _p_map(score_obs, null_max, True)
    sig = p_map < cfg.alpha
    return ClusterOutput(name, "pairwise_t", cfg.cluster_method.upper(), dataset.participants, sorted(set(dataset.groups)), [cond_a, cond_b], stat, score_obs, p_map, sig, null_max, dataset.times, dataset.ch_names, dataset.info, adj, _summarise_components(sig, stat, p_map, dataset.times, dataset.ch_names, adj, "t"), {"condition_a": cond_a, "condition_b": cond_b, "uses_bpm_difference": bpm_for_stat is not None, "df": int(df)})


def _plot_stat_map(result: ClusterOutput, out_file: Path, cfg: AnalysisConfig) -> None:
    stat = result.stat_obs
    vmax = np.nanpercentile(np.abs(stat), 99)
    vmax = 1.0 if not np.isfinite(vmax) or vmax <= 0 else vmax
    if result.kind == "omnibus_f":
        data, vmin, cmap, label = stat, 0.0, "viridis", "F"
    else:
        data, vmin, cmap, label = stat, -vmax, "RdBu_r", "t"
    fig_h = max(5.0, min(16.0, 0.05 * len(result.ch_names)))
    fig, ax = plt.subplots(figsize=(12, fig_h), constrained_layout=True)
    im = ax.imshow(data.T, origin="lower", aspect="auto", extent=[result.times[0], result.times[-1], -0.5, len(result.ch_names) - 0.5], vmin=vmin, vmax=vmax, cmap=cmap)
    if result.sig_mask.any():
        ax.contour(result.times, np.arange(len(result.ch_names)), result.sig_mask.T.astype(float), levels=[0.5], colors="black", linewidths=0.8)
    step = max(1, math.ceil(len(result.ch_names) / 36))
    yticks = np.arange(0, len(result.ch_names), step)
    ax.set_yticks(yticks)
    ax.set_yticklabels([result.ch_names[i] for i in yticks], fontsize=7)
    ax.set_xlabel("Time from R-peak (s)")
    ax.set_ylabel("EEG channel")
    ax.set_title(f"{result.name}: {label}-map, outline = corrected p < {cfg.alpha}")
    fig.colorbar(im, ax=ax).set_label(label)
    fig.savefig(out_file, dpi=cfg.fig_dpi)
    plt.close(fig)


def _plot_p_map(result: ClusterOutput, out_file: Path, cfg: AnalysisConfig) -> None:
    data = -np.log10(np.clip(result.p_map, 1e-12, 1.0))
    cmap = "viridis"
    vmin = 0.0
    vmax = np.nanpercentile(data, 99)
    if result.kind != "omnibus_f":
        data *= np.sign(result.stat_obs)
        vmax = np.nanpercentile(np.abs(data), 99)
        vmin = -vmax
        cmap = "RdBu_r"
    vmax = 1.0 if not np.isfinite(vmax) or vmax <= 0 else vmax
    fig, ax = plt.subplots(figsize=(12, max(5.0, min(16.0, 0.05 * len(result.ch_names)))), constrained_layout=True)
    im = ax.imshow(data.T, origin="lower", aspect="auto", extent=[result.times[0], result.times[-1], -0.5, len(result.ch_names) - 0.5], vmin=vmin, vmax=vmax, cmap=cmap)
    if result.sig_mask.any():
        ax.contour(result.times, np.arange(len(result.ch_names)), result.sig_mask.T.astype(float), levels=[0.5], colors="black", linewidths=0.8)
    ax.set_xlabel("Time from R-peak (s)")
    ax.set_ylabel("EEG channel")
    ax.set_title(f"{result.name}: corrected p-map")
    fig.colorbar(im, ax=ax).set_label("-log10(p)")
    fig.savefig(out_file, dpi=cfg.fig_dpi)
    plt.close(fig)


def _plot_peak_topomap(result: ClusterOutput, out_file: Path, cfg: AnalysisConfig) -> None:
    score = np.where(result.sig_mask, np.abs(result.stat_obs), -np.inf) if result.sig_mask.any() else np.abs(result.stat_obs)
    ti, ci = np.unravel_index(np.nanargmax(score), score.shape)
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    mne.viz.plot_topomap(result.stat_obs[ti], result.info, axes=ax, show=False, mask=result.sig_mask[ti] if result.sig_mask.any() else None, contours=0)
    label = "F" if result.kind == "omnibus_f" else "t"
    ax.set_title(f"{label}={result.stat_obs[ti, ci]:.3f}, p={result.p_map[ti, ci]:.4g}, t={result.times[ti]:.3f}s, {result.ch_names[ci]}")
    fig.savefig(out_file, dpi=cfg.fig_dpi)
    plt.close(fig)


def _plot_component_waves(dataset: ConditionDataset, result: ClusterOutput, out_dir: Path, cfg: AnalysisConfig) -> None:
    components = result.components.head(8)
    if components.empty:
        ti, ci = np.unravel_index(np.nanargmax(np.abs(result.stat_obs)), result.stat_obs.shape)
        components = pd.DataFrame([{"component": 0, "channels": result.ch_names[ci], "t_start": result.times[max(0, ti - 5)], "t_end": result.times[min(len(result.times) - 1, ti + 5)]}])
    for _, row in components.iterrows():
        ch_idx = [result.ch_names.index(ch) for ch in str(row["channels"]).split(";") if ch in result.ch_names]
        if not ch_idx:
            continue
        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
        if result.kind == "pairwise_t":
            ia, ib = dataset.conditions.index(result.conditions[0]), dataset.conditions.index(result.conditions[1])
            wave = (dataset.X[:, ia, :, :] - dataset.X[:, ib, :, :])[:, :, ch_idx].mean(axis=2) * 1e6
            ax.plot(dataset.times, wave.mean(axis=0), linewidth=2, label=f"{result.conditions[0]} - {result.conditions[1]}")
            ax.fill_between(dataset.times, wave.mean(axis=0) - wave.std(axis=0, ddof=1) / np.sqrt(len(wave)), wave.mean(axis=0) + wave.std(axis=0, ddof=1) / np.sqrt(len(wave)), alpha=0.2)
        else:
            for ci, cond in enumerate(dataset.conditions):
                wave = dataset.X[:, ci, :, :][:, :, ch_idx].mean(axis=2) * 1e6
                ax.plot(dataset.times, wave.mean(axis=0), linewidth=2, label=cond)
                ax.fill_between(dataset.times, wave.mean(axis=0) - wave.std(axis=0, ddof=1) / np.sqrt(len(wave)), wave.mean(axis=0) + wave.std(axis=0, ddof=1) / np.sqrt(len(wave)), alpha=0.15)
        ax.axhline(0.0, color="k", linewidth=0.8, alpha=0.4)
        ax.axvline(0.0, color="k", linewidth=0.8, alpha=0.4)
        ax.axvspan(float(row["t_start"]), float(row["t_end"]), alpha=0.12)
        ax.set_xlabel("Time from R-peak (s)")
        ax.set_ylabel("Component amplitude (uV)")
        ax.legend(frameon=False)
        fig.savefig(out_dir / f"component_{int(row['component']):02d}_waves.png", dpi=cfg.fig_dpi)
        plt.close(fig)


def _plot_paper_cluster_summary(dataset: ConditionDataset, result: ClusterOutput, out_dir: Path, cfg: AnalysisConfig) -> None:
    if not cfg.save_paper_style_plots:
        return
    components = result.components.head(6)
    if components.empty:
        components = pd.DataFrame([
            {
                "component": 0,
                "channels": result.ch_names[int(np.nanargmax(np.nanmax(np.abs(result.stat_obs), axis=0)))],
                "t_start": float(result.times[0]),
                "t_end": float(result.times[-1]),
                "min_p": float(np.nanmin(result.p_map)),
            }
        ])
    for _, row in components.iterrows():
        ch_idx = [result.ch_names.index(ch) for ch in str(row["channels"]).split(";") if ch in result.ch_names]
        if not ch_idx:
            continue
        t_mask = (result.times >= float(row["t_start"])) & (result.times <= float(row["t_end"]))
        topo = np.nanmean(result.stat_obs[t_mask], axis=0) if t_mask.any() else np.nanmean(result.stat_obs, axis=0)
        fig = plt.figure(figsize=(10, 4), constrained_layout=True)
        gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1.0])
        ax_wave = fig.add_subplot(gs[0, 0])
        ax_topo = fig.add_subplot(gs[0, 1])
        if result.kind == "pairwise_t":
            ia, ib = dataset.conditions.index(result.conditions[0]), dataset.conditions.index(result.conditions[1])
            waves = []
            for ci, cond in [(ia, result.conditions[0]), (ib, result.conditions[1])]:
                wave = dataset.X[:, ci, :, :][:, :, ch_idx].mean(axis=2) * 1e6
                waves.append(wave)
                mean = wave.mean(axis=0)
                sem = wave.std(axis=0, ddof=1) / np.sqrt(wave.shape[0])
                ax_wave.plot(dataset.times, mean, linewidth=2, label=cond)
                ax_wave.fill_between(dataset.times, mean - sem, mean + sem, alpha=0.16)
        else:
            for ci, cond in enumerate(dataset.conditions):
                wave = dataset.X[:, ci, :, :][:, :, ch_idx].mean(axis=2) * 1e6
                mean = wave.mean(axis=0)
                sem = wave.std(axis=0, ddof=1) / np.sqrt(wave.shape[0])
                ax_wave.plot(dataset.times, mean, linewidth=2, label=cond)
                ax_wave.fill_between(dataset.times, mean - sem, mean + sem, alpha=0.16)
        ax_wave.axhline(0, color="k", linewidth=0.8, alpha=0.35)
        ax_wave.axvline(0, color="k", linewidth=0.8, alpha=0.35)
        ax_wave.axvspan(float(row["t_start"]), float(row["t_end"]), color="0.75", alpha=0.35)
        ax_wave.set_xlabel("Time from R-peak (s)")
        ax_wave.set_ylabel("Amplitude (uV)")
        ax_wave.legend(frameon=False)
        ax_wave.set_title(f"Component {int(row['component'])}: mean +/- SEM")
        mask = np.zeros(len(result.ch_names), dtype=bool)
        mask[ch_idx] = True
        mne.viz.plot_topomap(topo, result.info, axes=ax_topo, show=False, mask=mask, contours=0)
        ax_topo.set_title(f"Average statistic\n{row['t_start']:.3f}-{row['t_end']:.3f}s")
        fig.savefig(out_dir / f"paper_component_{int(row['component']):02d}_wave_topomap.png", dpi=cfg.fig_dpi)
        plt.close(fig)


def _plot_null_histogram(result: ClusterOutput, out_file: Path, cfg: AnalysisConfig) -> None:
    if len(result.null_max) == 0:
        return
    obs_components = result.components
    obs_values = []
    if not obs_components.empty:
        score_label = "F" if result.kind == "omnibus_f" else "t"
        peak_col = f"peak_{score_label}"
        for _, row in obs_components.iterrows():
            # Prefer the actual cluster score from score_obs within the component window/channel set.
            channels = [ch for ch in str(row["channels"]).split(";") if ch in result.ch_names]
            ch_idx = [result.ch_names.index(ch) for ch in channels]
            t_mask = (result.times >= float(row["t_start"])) & (result.times <= float(row["t_end"]))
            if ch_idx and t_mask.any():
                vals = result.score_obs[np.ix_(t_mask, ch_idx)]
                obs_values.append(float(np.nanmax(np.abs(vals))))
            elif peak_col in row:
                obs_values.append(abs(float(row[peak_col])))
    else:
        obs_values = [float(np.nanmax(np.abs(result.score_obs)))]
    cutoff = float(np.percentile(result.null_max, 95))
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.hist(result.null_max, bins=30, color="0.65", edgecolor="white")
    ax.axvline(cutoff, color="#1f77b4", linestyle="--", linewidth=2, label="95% cutoff")
    for i, value in enumerate(obs_values[:8]):
        ax.axvline(value, color="#c62828", linewidth=1.8, alpha=0.9, label="observed cluster" if i == 0 else None)
    ax.set_xlabel(f"Maximum |{_score_label(cfg)}|")
    ax.set_ylabel("Count")
    ax.set_title(f"{result.name}: permutation/control distribution")
    ax.legend(frameon=False)
    fig.savefig(out_file, dpi=cfg.fig_dpi)
    plt.close(fig)


def _save_result(result: ClusterOutput, dataset: ConditionDataset, cfg: AnalysisConfig) -> None:
    out = cfg.output_dir / safe_name(result.name)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "cluster_results.npz", stat_obs=result.stat_obs, score_obs=result.score_obs, p_map=result.p_map, sig_mask=result.sig_mask, null_max=result.null_max, times=result.times, ch_names=np.array(result.ch_names, dtype=object))
    result.components.to_csv(out / "components.csv", index=False)
    ti, ci = np.where(result.sig_mask)
    pd.DataFrame([{"time": float(result.times[t]), "channel": result.ch_names[c], "statistic": float(result.stat_obs[t, c]), "p": float(result.p_map[t, c])} for t, c in zip(ti, ci)]).to_csv(out / "significant_points.csv", index=False)
    _plot_stat_map(result, out / "stat_map_with_sig_outline.png", cfg)
    _plot_p_map(result, out / "corrected_p_map.png", cfg)
    _plot_peak_topomap(result, out / "peak_topomap.png", cfg)
    _plot_component_waves(dataset, result, out, cfg)
    _plot_paper_cluster_summary(dataset, result, out, cfg)
    _plot_null_histogram(result, out / "paper_null_distribution.png", cfg)
    summary = {
        "name": result.name,
        "kind": result.kind,
        "method": result.method,
        "n_participants": len(result.participants),
        "conditions": result.conditions,
        "control_mode": cfg.control_mode,
        "n_significant_samples": int(result.sig_mask.sum()),
        "n_components": int(len(result.components)),
        "extra": result.extra,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _save_grand_averages(dataset: ConditionDataset, cfg: AnalysisConfig) -> None:
    out = cfg.output_dir / f"grand_averages_{safe_name(dataset.name)}"
    out.mkdir(parents=True, exist_ok=True)
    template = mne.EvokedArray(dataset.X[0, 0].T, dataset.info, tmin=float(dataset.times[0]), comment="template")
    for ci, cond in enumerate(dataset.conditions):
        ev = template.copy()
        ev.data = dataset.X[:, ci].mean(axis=0).T
        ev.nave = dataset.X.shape[0]
        ev.comment = f"grand_average:{cond}"
        ev.save(out / f"grand_average_{safe_name(cond)}-ave.fif", overwrite=True)
        try:
            fig = ev.plot_joint(times=np.linspace(cfg.tmin, cfg.tmax, 5), show=False, title=f"Grand average: {cond}")
            fig.savefig(out / f"grand_average_{safe_name(cond)}_joint.png", dpi=cfg.fig_dpi)
            plt.close(fig)
        except Exception as exc:
            warnings.warn(f"Could not plot grand average for {cond}: {exc}")


def run_analysis_from_manifest(cfg: AnalysisConfig) -> dict[str, object]:
    cfg.manifest_csv = Path(cfg.manifest_csv)
    cfg.output_dir = Path(cfg.output_dir)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    if len(cfg.conditions) < 2:
        raise ValueError("At least two conditions are required.")
    df = _read_manifest(cfg)
    df.to_csv(cfg.output_dir / "input_manifest_resolved.csv", index=False)
    loaded = _load_rows(df, cfg)
    pd.DataFrame([{"row_index": r.row_index, "participant": r.participant, "group": r.group, "condition": r.condition, "file_path": str(r.file_path), "kind": r.kind, "nave": r.nave, "n_epochs": r.n_epochs, "bpm": r.bpm} for r in loaded]).to_csv(cfg.output_dir / "loaded_rows_summary.csv", index=False)
    datasets: list[ConditionDataset] = [_build_dataset(loaded, cfg, "all_participants")]
    if cfg.run_within_groups:
        for group in sorted({r.group for r in loaded}):
            try:
                datasets.append(_build_dataset(loaded, cfg, f"within_group_{safe_name(group)}", group_filter=group))
            except RuntimeError as exc:
                warnings.warn(f"Skipping group {group!r}: {exc}")
    results = []
    for dataset in datasets:
        _save_grand_averages(dataset, cfg)
        if cfg.run_omnibus and len(dataset.conditions) >= 3:
            results.append((_run_omnibus(dataset, cfg, f"{dataset.name}_condition_main_effect"), dataset))
        if cfg.run_pairwise:
            for cond_a, cond_b in combinations(dataset.conditions, 2):
                results.append((_run_pairwise(dataset, cfg, cond_a, cond_b, f"{dataset.name}_pairwise_{safe_name(cond_a)}_minus_{safe_name(cond_b)}"), dataset))
    for result, dataset in results:
        _save_result(result, dataset, cfg)
    manifest = {
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items()},
        "tests_run": [r.name for r, _ in results],
        "mne_version": mne.__version__,
    }
    (cfg.output_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "output_dir": str(cfg.output_dir),
        "tests_run": [r.name for r, _ in results],
        "summaries": [
            {
                "name": r.name,
                "kind": r.kind,
                "method": r.method,
                "n_participants": len(r.participants),
                "n_significant_samples": int(r.sig_mask.sum()),
                "n_components": int(len(r.components)),
            }
            for r, _ in results
        ],
    }
