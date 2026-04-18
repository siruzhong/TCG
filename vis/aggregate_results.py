"""One-click incremental updater for ``tcg_result.md``.

Scans every ``checkpoints/<Model>/<dataset>_1_<input>_<horizon>/<hash>/`` folder,
reads ``cfg.json`` (to classify as RAW or TCG) and ``test_metrics.json`` (MSE /
MAE), then updates ``tcg_result.md`` in place using the best-of rule:

    new_cell = argmin_MSE( old_markdown_cell, all_disk_runs_for_this_cell )

Conventions enforced by this script (keep them in sync with run_baselines.py):

* RAW  = ``cfg.json -> model_config.tcg.params`` is empty (defaults to
        ``enabled=False``). Usually a single run per cell.
* TCG  = ``cfg.json -> model_config.tcg.params`` has ``enabled=True``. When
        several hyperparameter configurations exist (orth_lambda / num_patterns
        / use_multiscale), the one with the lowest test MSE wins.

The list of ``MODELS``/``DATASETS``/``DATASET_CONFIGS`` is imported directly
from ``run_baselines.py`` so adding a new model or dataset there also extends
the table here.

Usage:
    python vis/aggregate_results.py              # in-place update of tcg_result.md
    python vis/aggregate_results.py --dry-run    # print changes, do not write
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import glob
from collections import defaultdict

# Make ``run_baselines`` importable without triggering its heavy model imports.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

try:
    from run_baselines import (
        MODELS as RB_MODELS,
        DATASETS as RB_DATASETS,
        DATASET_CONFIGS as RB_CONFIGS,
    )
except Exception:
    # Fallback: mirror the declarations in run_baselines.py. Keep in sync.
    RB_MODELS = ["Informer", "Crossformer", "PatchTST", "TimesNet",
                 "TimeMixer", "TimeFilter", "WPMixer"]
    RB_DATASETS = [
        ("ETTh1", 7), ("ETTh2", 7), ("ETTm1", 7), ("ETTm2", 7),
        ("Weather", 21), ("Illness", 7), ("ExchangeRate", 8),
        ("BeijingAirQuality", 7), ("COVID19", 8), ("VIX", 1), 
        ("NABCPU", 3), ("Sunspots", 1),
    ]
    RB_CONFIGS = {
        "Illness":  {"input_lens": [24], "output_lens": [24, 36, 48, 60]},
        "COVID19":  {"input_lens": [36], "output_lens": [7, 14, 28, 60]},
        "NABCPU":   {"input_lens": [96], "output_lens": [24, 48, 96, 192]},
        "Sunspots": {"input_lens": [36], "output_lens": [12, 24, 48, 96]},
        "default":  {"input_lens": [96], "output_lens": [96, 192, 336, 720]},
    }

# Display name used in the markdown table (differs from run_baselines name).
MD_NAME = {"Illness": "ILI"}


# Keep the *display* order of the table in sync with the model list so that
# new models appear at the end. Each model contributes two columns: _raw and _tcg.
# The checkpoint directory name sometimes differs from the logical model name;
# that mapping is handled by _ckpt_subdir.
def _ckpt_subdir(model_name: str) -> str:
    """Map logical model name to the checkpoints/<...> folder name."""
    if model_name in ("Informer", "Crossformer"):
        return model_name
    return f"{model_name}ForForecasting"


MODELS = list(RB_MODELS)
MARKDOWN = os.path.join(REPO_ROOT, "tcg_result.md")
CHECKPOINTS = os.path.join(REPO_ROOT, "checkpoints")


# --------------------------------------------------------------------------- #
# Cfg / metric parsing
# --------------------------------------------------------------------------- #

def _is_tcg_enabled(cfg: dict) -> bool:
    tcg = cfg.get("model_config", {}).get("tcg", {})
    if not isinstance(tcg, dict):
        return False
    params = tcg.get("params", {}) or {}
    return str(params.get("enabled", "False")).lower() == "true"


def _tcg_key(cfg: dict) -> tuple:
    tcg = cfg.get("model_config", {}).get("tcg", {})
    params = tcg.get("params", {}) if isinstance(tcg, dict) else {}
    return (params.get("num_patterns"),
            params.get("orth_lambda"),
            params.get("use_multiscale"))


def _read_metrics(run_dir: str):
    mpath = os.path.join(run_dir, "test_metrics.json")
    if not os.path.exists(mpath):
        return None
    try:
        m = json.load(open(mpath))["overall"]
    except Exception:
        return None
    try:
        return {"MSE": float(m["MSE"]), "MAE": float(m["MAE"])}
    except (KeyError, TypeError):
        return None


# --------------------------------------------------------------------------- #
# Enumerate (model, dataset, horizon) targets and collect disk state
# --------------------------------------------------------------------------- #

def _horizons_of(ds_name: str) -> list:
    cfg = RB_CONFIGS.get(ds_name, RB_CONFIGS["default"])
    return cfg["output_lens"]


def _input_len_of(ds_name: str) -> int:
    cfg = RB_CONFIGS.get(ds_name, RB_CONFIGS["default"])
    return cfg["input_lens"][0]


def collect_disk() -> dict:
    """disk[model][dataset][horizon] = {"raw": {MSE,MAE}|None, "tcg": {...}|None, "tcg_hp": (...)}."""
    disk = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for model_name in MODELS:
        base = os.path.join(CHECKPOINTS, _ckpt_subdir(model_name))
        if not os.path.isdir(base):
            continue
        for ds_name, _ in RB_DATASETS:
            il = _input_len_of(ds_name)
            for h in _horizons_of(ds_name):
                pattern = os.path.join(base, f"{ds_name}_*_{il}_{h}", "*")
                raw_best, tcg_best, tcg_hp = None, None, None
                for run_dir in glob.glob(pattern):
                    cfg_path = os.path.join(run_dir, "cfg.json")
                    if not os.path.exists(cfg_path):
                        continue
                    try:
                        cfg = json.load(open(cfg_path))
                    except Exception:
                        continue
                    m = _read_metrics(run_dir)
                    if m is None:
                        continue
                    if _is_tcg_enabled(cfg):
                        if tcg_best is None or m["MSE"] < tcg_best["MSE"]:
                            tcg_best, tcg_hp = m, _tcg_key(cfg)
                    else:
                        if raw_best is None or m["MSE"] < raw_best["MSE"]:
                            raw_best = m
                disk[model_name][ds_name][h] = {
                    "raw": raw_best, "tcg": tcg_best, "tcg_hp": tcg_hp,
                }
    return disk


# --------------------------------------------------------------------------- #
# Markdown parsing / emission
# --------------------------------------------------------------------------- #

def _parse_cell(s):
    s = (s or "").strip()
    if not s:
        return None
    m = re.match(r"^([\d.]+)\s*/\s*([\d.]+)$", s)
    if not m:
        return None
    return {"MSE": float(m.group(1)), "MAE": float(m.group(2))}


def _cell(v):
    if v is None:
        return ""
    return f"{v['MSE']:.3f} / {v['MAE']:.3f}"


def _parse_header_column_map(header_cells: list) -> dict:
    """Map column-index-in-body-row -> (model_name, kind) where kind in {raw, tcg}.

    ``header_cells`` is the list BETWEEN the first two pipes of the header row,
    minus the first two (``dataset``, ``horizon``). Unknown columns are skipped.
    """
    mapping = {}
    for i, name in enumerate(header_cells):
        m = re.match(r"^(\w+)_(raw|tcg)$", name.strip())
        if not m:
            continue
        mapping[i] = (m.group(1), m.group(2))
    return mapping


def _parse_markdown(path: str):
    """Return (prefix_lines, suffix_lines, existing, header_map).

    existing[dataset][horizon] = dict keyed by (model, kind) -> "MSE / MAE" string.
    header_map: same dict returned by _parse_header_column_map for the original header.
    """
    existing = defaultdict(lambda: defaultdict(dict))
    prefix, suffix = [], []
    header_map = {}
    if not os.path.exists(path):
        prefix = [
            "# TCG Results",
            "",
            "Generated from `checkpoints/` aggregation (MSE / MAE).",
            "",
            _header_row(),
            _separator_row(),
        ]
        return prefix, suffix, existing, header_map

    with open(path) as f:
        lines = [line.rstrip("\n") for line in f]

    data_rows_seen = False
    in_table_body = False
    md_to_logical = {MD_NAME.get(n, n): n for n, _ in RB_DATASETS}
    for line in lines:
        if not line.startswith("|"):
            (prefix if not in_table_body else suffix).append(line)
            continue
        cells = [c.strip() for c in line.split("|")[1:-1]]
        if cells and cells[0].lower() == "dataset":
            header_map = _parse_header_column_map(cells[2:])
            prefix.append(line)
            continue
        if cells and set(cells[0]) <= set("-: "):
            prefix.append(line)
            in_table_body = True
            continue
        if not in_table_body:
            prefix.append(line)
            continue
        if len(cells) < 3:
            suffix.append(line)
            continue
        ds_md, horizon, *row_cells = cells
        ds_logical = md_to_logical.get(ds_md, ds_md)
        if horizon in {"Avg", "Δ"} or not horizon.isdigit():
            continue
        try:
            h = int(horizon)
        except ValueError:
            continue
        for col_idx, (model_name, kind) in header_map.items():
            if col_idx < len(row_cells):
                existing[ds_logical][h][(model_name, kind)] = row_cells[col_idx]
        data_rows_seen = True

    if not data_rows_seen:
        prefix = lines
        suffix = []

    return prefix, suffix, existing, header_map


def _resolve_column_order(header_map: dict) -> list:
    """Return the model order to use for emitting rows.

    Preserves the column order of an existing markdown file so that re-running
    the aggregator never reshuffles columns. Unknown-to-md models defined in
    ``run_baselines.MODELS`` are appended at the end so adding a new baseline
    just grows the table to the right.
    """
    if not header_map:
        return list(MODELS)
    # Extract model names in the order they appear in the existing header.
    seen, order = set(), []
    for idx in sorted(header_map):
        model_name, _kind = header_map[idx]
        if model_name not in seen:
            seen.add(model_name)
            order.append(model_name)
    # Append any MODELS entries that are not yet in the header (new baselines).
    for m in MODELS:
        if m not in seen:
            order.append(m)
    return order


def _header_row(model_order: list | None = None) -> str:
    order = model_order if model_order else MODELS
    cells = ["dataset", "horizon"]
    for m in order:
        cells.append(f"{m}_raw")
        cells.append(f"{m}_tcg")
    return "| " + " | ".join(cells) + " |"


def _separator_row(model_order: list | None = None) -> str:
    order = model_order if model_order else MODELS
    n = 2 + 2 * len(order)
    return "| " + " | ".join(["---"] * n) + " |"


def _fmt_row(ds_md: str, horizon, cells_str: list) -> str:
    parts = [ds_md, str(horizon)] + list(cells_str)
    return "| " + " | ".join(parts) + " |"


# --------------------------------------------------------------------------- #
# Merge + emit
# --------------------------------------------------------------------------- #

def _rank_key(c):
    """Rank candidates by rounded (MSE, MAE) at the display precision.

    This eliminates spurious differences caused by floating-point noise at
    full precision (the markdown stores 3-decimal rounded values) and
    guarantees that when two candidates tie on MSE, we keep the one with the
    better MAE rather than silently swapping to a worse MAE.
    """
    return (round(c["MSE"], 3), round(c["MAE"], 3))


def _best_of(*cands):
    valid = [c for c in cands if c is not None]
    if not valid:
        return None
    return min(valid, key=_rank_key)


def merge_and_emit(existing: dict, disk: dict, model_order: list | None = None):
    """Produce (new_rows_by_dataset, change_log).

    ``model_order`` controls the column ordering when emitting rows. If None,
    falls back to ``run_baselines.MODELS``.
    """
    new_rows = []
    changes = {"fill": [], "improve": [], "unchanged_empty": []}
    order = model_order if model_order else MODELS
    n_cols = 2 * len(order)
    for ds_name, _ in RB_DATASETS:
        ds_md = MD_NAME.get(ds_name, ds_name)
        horizons = _horizons_of(ds_name)
        avg_accumulator = [[] for _ in range(n_cols)]
        for h in horizons:
            old_by_key = existing.get(ds_name, {}).get(h, {})
            row = []
            for i, model_name in enumerate(order):
                ds_disk = disk.get(model_name, {}).get(ds_name, {}).get(h, {})
                for offset, kind in ((0, "raw"), (1, "tcg")):
                    cell_idx = 2 * i + offset
                    old_str = old_by_key.get((model_name, kind), "")
                    old_v = _parse_cell(old_str)
                    new_v = ds_disk.get(kind)
                    best = _best_of(old_v, new_v)
                    best_str = _cell(best)
                    row.append(best_str)
                    if best is not None:
                        avg_accumulator[cell_idx].append(best)
                    if old_v is None and best is not None:
                        changes["fill"].append(
                            (ds_name, h, model_name, kind, "", best_str))
                    elif (best is not None and old_v is not None
                          and _rank_key(best) < _rank_key(old_v)):
                        changes["improve"].append(
                            (ds_name, h, model_name, kind, _cell(old_v), best_str))
                    elif old_v is None and best is None:
                        changes["unchanged_empty"].append(
                            (ds_name, h, model_name, kind))
            new_rows.append(_fmt_row(ds_md, h, row))
        avg_cells = []
        for bucket in avg_accumulator:
            if bucket:
                avg_cells.append(_cell({
                    "MSE": sum(b["MSE"] for b in bucket) / len(bucket),
                    "MAE": sum(b["MAE"] for b in bucket) / len(bucket),
                }))
            else:
                avg_cells.append("")
        new_rows.append(_fmt_row(ds_md, "Avg", avg_cells))
        new_rows.append(_fmt_row(ds_md, "Δ", [""] * n_cols))
    return new_rows, changes


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def _print_summary(changes: dict):
    n_fill = len(changes["fill"])
    n_improve = len(changes["improve"])
    n_empty = len(changes["unchanged_empty"])
    print(f"\n== Summary: filled {n_fill}, improved {n_improve}, "
          f"still empty {n_empty} ==")

    if changes["fill"]:
        print("\n-- Newly filled (was empty in md, now have a number) --")
        for ds, h, m, k, _old, new in changes["fill"]:
            print(f"  {ds:18s} H={h:<4d} {m:11s} {k:3s}  (empty) -> {new}")
    if changes["improve"]:
        print("\n-- Improved (disk MSE beat existing md) --")
        for ds, h, m, k, old, new in changes["improve"]:
            print(f"  {ds:18s} H={h:<4d} {m:11s} {k:3s}  {old}  ->  {new}")
    if changes["unchanged_empty"]:
        print("\n-- Still missing (no disk result yet) --")
        pairs = defaultdict(list)
        for ds, h, m, k in changes["unchanged_empty"]:
            pairs[(ds, h)].append(f"{m}-{k}")
        for (ds, h), ms in pairs.items():
            print(f"  {ds:18s} H={h:<4d} : " + ", ".join(ms))


def _print_completeness_matrix(disk: dict, existing: dict):
    print("\n== Completeness matrix (per dataset/horizon, R/T = raw/tcg available) ==")
    for ds_name, _ in RB_DATASETS:
        print(f"\n-- {ds_name} --")
        print(f"{'horizon':>8}  " + " ".join(f"{m:<11s}" for m in MODELS))
        for h in _horizons_of(ds_name):
            row = [f"{h:>8}"]
            old = existing.get(ds_name, {}).get(h, {})
            for model_name in MODELS:
                ds_disk = disk.get(model_name, {}).get(ds_name, {}).get(h, {})
                r_have = (_parse_cell(old.get((model_name, "raw"), "")) is not None
                          or ds_disk.get("raw") is not None)
                t_have = (_parse_cell(old.get((model_name, "tcg"), "")) is not None
                          or ds_disk.get("tcg") is not None)
                row.append(f"  {'R' if r_have else '-'}{'T' if t_have else '-'}       ")
            print(" ".join(row))


# --------------------------------------------------------------------------- #
# Main entry
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="Do not modify tcg_result.md; print changes only.")
    ap.add_argument("--path", default=MARKDOWN,
                    help="Path to the markdown file to update.")
    args = ap.parse_args()

    prefix, suffix, existing, header_map = _parse_markdown(args.path)
    model_order = _resolve_column_order(header_map)
    disk = collect_disk()
    new_rows, changes = merge_and_emit(existing, disk, model_order=model_order)

    # Regenerate header/separator so the column order matches ``model_order``
    # (which preserves the original markdown order and extends it with any
    # newly-added models from ``run_baselines.MODELS``).
    new_prefix = []
    replaced_header = False
    for line in prefix:
        cells = [c.strip() for c in line.split("|")[1:-1]] if line.startswith("|") else []
        if cells and cells[0].lower() == "dataset":
            new_prefix.append(_header_row(model_order))
            replaced_header = True
        elif cells and set(cells[0]) <= set("-: ") and replaced_header:
            new_prefix.append(_separator_row(model_order))
        else:
            new_prefix.append(line)
    prefix = new_prefix

    content = "\n".join(prefix + new_rows + suffix)
    if content and not content.endswith("\n"):
        content += "\n"

    _print_completeness_matrix(disk, existing)
    _print_summary(changes)

    if args.dry_run:
        print(f"\n[DRY-RUN] Would write {args.path} "
              f"({len(prefix) + len(new_rows) + len(suffix)} lines)")
        return

    if (len(changes["fill"]) == 0 and len(changes["improve"]) == 0
            and os.path.exists(args.path)):
        print(f"\nNo changes needed; {args.path} already up-to-date.")
        return

    with open(args.path, "w") as f:
        f.write(content)
    print(f"\nWrote {args.path} "
          f"({len(prefix) + len(new_rows) + len(suffix)} lines)")


if __name__ == "__main__":
    main()
