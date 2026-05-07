#!/usr/bin/env python3
"""
Extract parameter counts (and optionally FLOPs) for the scaling vs DPR table.

1) checkpoints/test_scaling — reads each run's training_log*.log line:
     Total parameters: <N>
   and cfg.json to label 2xW / 2xD / 2xB (matches run_rq3_scaling.WIDE_CONFIGS).

2) Raw and +DPR are usually NOT under test_scaling (RQ3 scaling driver only runs WIDE).
   This script *computes* their param counts (and optional FLOPs) from the
   same configs as run_rq3_scaling.get_model_config for ETTh1 96->96
   (change TASK_REF below if needed).

FLOPs: not logged anywhere in BasicTS. If ``pip install thop`` is available,
  pass --flops to estimate forward MACs (shown as GMac in thop output).

Usage (from repository root):
  python scripts/extract_scaling_costs.py
  python scripts/extract_scaling_costs.py --flops
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from glob import glob
from typing import Any, Dict, Optional, Tuple

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
_SRC = os.path.join(_PROJECT_ROOT, "src")
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

TEST_SCALING = os.path.join(_PROJECT_ROOT, "checkpoints", "test_scaling")

# Match run_rq3_scaling.WIDE_CONFIGS
WIDE_RULES: Tuple[Dict[str, Any], ...] = (
    {
        "label": "2xW",
        "tag": "w2_d1",
        "hidden_size": 512,
        "intermediate_size": 2048,
        "num_layers": 1,
    },
    {
        "label": "2xD",
        "tag": "w1_d2",
        "hidden_size": 256,
        "intermediate_size": 1024,
        "num_layers": 2,
    },
    {
        "label": "2xB",
        "tag": "w2_d2",
        "hidden_size": 512,
        "intermediate_size": 2048,
        "num_layers": 2,
    },
)

# Reference task for "cost row" (table uses one hardware profile per arch).
TASK_REF = ("ETTh1", 7, 96, 96)


def _fix_json_text(s: str) -> str:
    return (
        s.replace(": NaN", ": null")
        .replace(": Infinity", ': "Infinity"')
        .replace(": -Infinity", ': "-Infinity"')
    )


def _parse_log_params(log_path: str) -> Optional[int]:
    pat = re.compile(r"Total parameters:\s*(\d+)")
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = pat.search(line)
            if m:
                return int(m.group(1))
    return None


def _short_model(name: str) -> str:
    for s in ("PatchTST", "TimesNet", "TimeMixer"):
        if s in name:
            return s
    return name


def _label_wide(mc: dict) -> Optional[str]:
    hs = mc.get("hidden_size")
    nl = mc.get("num_layers")
    inter = mc.get("intermediate_size")
    for r in WIDE_RULES:
        if hs == r["hidden_size"] and nl == r["num_layers"] and inter == r["intermediate_size"]:
            return r["label"]
    return None


def _load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.loads(_fix_json_text(f.read()))


def collect_test_scaling() -> list[dict]:
    rows = []
    if not os.path.isdir(TEST_SCALING):
        return rows
    for name in os.listdir(TEST_SCALING):
        sub = os.path.join(TEST_SCALING, name)
        if not os.path.isdir(sub):
            continue
        cfgp = os.path.join(sub, "cfg.json")
        logs = glob(os.path.join(sub, "training_log*.log"))
        if not os.path.isfile(cfgp) or not logs:
            continue
        try:
            cfg = _load_cfg(cfgp)
        except (json.JSONDecodeError, OSError):
            continue
        mc = cfg.get("model_config") or {}
        if not isinstance(mc, dict):
            continue
        wide = _label_wide(mc)
        if not wide:
            continue
        nparams = _parse_log_params(logs[0])
        if nparams is None:
            continue
        mname = cfg.get("model", {})
        if isinstance(mname, dict):
            mname = mname.get("name", "")
        rows.append(
            {
                "md5": name,
                "model": _short_model(str(mname)),
                "dataset": cfg.get("dataset_name"),
                "input_len": mc.get("input_len"),
                "output_len": mc.get("output_len"),
                "wide_label": wide,
                "num_features": mc.get("num_features"),
                "nparams": nparams,
            }
        )
    return rows


def _get_rq3_scaling_config(model_short: str, dataset_name, num_f, in_l, out_l, overrides, dpr_enabled: Optional[bool]):
    import run_rq3_scaling as rq3
    from basicts.configs import DPRConfig

    model_name = (
        "PatchTST"
        if model_short == "PatchTST"
        else "TimesNet"
        if model_short == "TimesNet"
        else "TimeMixer"
    )
    cls, mcfg, _ = rq3.get_model_config(
        model_name, in_l, out_l, num_f, dataset_name, overrides=overrides
    )
    if hasattr(mcfg, "dpr") and dpr_enabled is not None:
        mcfg.dpr = DPRConfig(enabled=dpr_enabled) if dpr_enabled else DPRConfig(enabled=False)
    return cls, mcfg


def count_params_for_config(model_class, model_config) -> int:
    m = model_class(model_config)
    return sum(p.numel() for p in m.parameters())


def flops_thop(model_class, model_config, model_short: str) -> Optional[float]:
    try:
        from thop import profile
        import torch
    except ImportError:
        return None

    mc = model_config
    in_l, out_l, nf = mc.input_len, mc.output_len, mc.num_features
    device = "cpu"
    x = torch.randn(1, in_l, nf, device=device)
    m = model_class(model_config).to(device)
    m.eval()
    if model_short == "TimesNet":
        # shape [B, L, n_ts] — length matches forward in dataloader; use 4 as in get_timestamp_sizes ETTh1
        n_ts = 4
        ts = torch.zeros(1, in_l, n_ts, device=device, dtype=torch.long)
        macs, _ = profile(m, inputs=(x, ts), verbose=False)
    else:
        macs, _ = profile(m, inputs=(x,), verbose=False)
    return float(macs) * 1e-9  # GMac (thop reports MACs as default)


def build_raw_dpr_params_and_flops(compute_flops: bool) -> Dict[str, Any]:
    """Per-architecture counts for TASK_REF, default width (Raw) and DPR on (+DPR)."""
    dname, nf, in_l, out_l = TASK_REF
    out: Dict[str, Any] = {}
    for model_short in ("PatchTST", "TimesNet", "TimeMixer"):
        # Raw: no RQ3 scaling overrides, DPR off (as in run_rq3_scaling.run_experiment for WIDE, but without overrides = defaults)
        cls_r, mcfg_r = _get_rq3_scaling_config(model_short, dname, nf, in_l, out_l, None, dpr_enabled=False)
        pr = count_params_for_config(cls_r, mcfg_r)
        fr = flops_thop(cls_r, mcfg_r, model_short) if compute_flops else None

        # +DPR: same backbone, DPR on (use patch-friendly defaults like run_rq2_baselines)
        from basicts.configs import DPRConfig

        cls_t, mcfg_t = _get_rq3_scaling_config(model_short, dname, nf, in_l, out_l, None, dpr_enabled=None)
        use_ms = model_short not in {"PatchTST", "WPMixer", "TimeFilter"}
        mcfg_t.dpr = DPRConfig(enabled=True, num_patterns=8, orth_lambda=0.01, use_multiscale=use_ms)
        pt = count_params_for_config(cls_t, mcfg_t)
        ft = flops_thop(cls_t, mcfg_t, model_short) if compute_flops else None

        out[model_short] = {
            "raw_params": pr,
            "dpr_params": pt,
            "raw_gmac": fr,
            "dpr_gmac": ft,
        }
    return out


def _fmt_ratio(x: float, base: float) -> str:
    r = x / base if base else 0.0
    if abs(r - round(r, 1)) < 0.01 and abs(r) < 100:
        return f"{r:.1f}×"
    if abs(r - round(r, 2)) < 0.01:
        return f"{r:.2f}×"
    return f"{r:.3g}×"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flops", action="store_true", help="Also estimate FLOPs (needs: pip install thop)")
    args = ap.parse_args()

    rows = collect_test_scaling()
    if not rows:
        print("No valid runs under", TEST_SCALING, file=sys.stderr)
    else:
        print("=== From checkpoints/test_scaling training logs: Total parameters ===\n")
        for model in ("PatchTST", "TimesNet", "TimeMixer"):
            for label in ("2xW", "2xD", "2xB"):
                # pick ETTh1 96-96 for display (cost row)
                mrows = [
                    r
                    for r in rows
                    if r["model"] == model
                    and r["wide_label"] == label
                    and r.get("dataset") == "ETTh1"
                    and r.get("input_len") == 96
                ]
                if not mrows:
                    mrows = [r for r in rows if r["model"] == model and r["wide_label"] == label]
                if mrows:
                    r0 = mrows[0]
                    print(
                        f"  {model:10s} {label:4s}  nparams={r0['nparams']:,}  (log: {r0['md5']}/training_log*.log)"
                    )
                else:
                    print(f"  {model:10s} {label:4s}  (no matching run)")

    print()
    if args.flops:
        try:
            import thop  # noqa: F401
        except ImportError:
            print(
                "FLOPs: install thop first:  pip install thop\n"
                "      (FLOPs are not written to training logs in this codebase.)"
            )
            args.flops = False

    print("=== Raw / +DPR: parameter counts (instantiated from config, not from logs) ===\n")
    os.chdir(_PROJECT_ROOT)
    stats = build_raw_dpr_params_and_flops(compute_flops=args.flops)
    dname, nf, in_l, out_l = TASK_REF
    print(f"Reference task: {dname}  ({in_l} -> {out_l})  num_features={nf}\n")
    for model, d in stats.items():
        print(f"  {model}:")
        print(
            f"    Raw:   {d['raw_params']:,} params"
            + (f"  |  ~{d['raw_gmac']:.3f} GMac" if d.get("raw_gmac") is not None else "")
        )
        print(
            f"    +DPR:  {d['dpr_params']:,} params"
            + (f"  |  ~{d['dpr_gmac']:.3f} GMac" if d.get("dpr_gmac") is not None else "")
        )
        if d["raw_params"]:
            print(f"    +DPR / Raw  params: {_fmt_ratio(d['dpr_params'], d['raw_params'])}")
        if args.flops and d.get("raw_gmac") and d.get("dpr_gmac"):
            print(f"    +DPR / Raw  FLOPs:  {_fmt_ratio(d['dpr_gmac'], d['raw_gmac'])}")
        print()

    # Relative Params row (for LaTeX), vs same-arch Raw on TASK_REF
    print("=== LaTeX-friendly relative param multipliers (vs Raw), ETTh1 96→96 ===\n")
    log_map: Dict[str, Dict[str, int]] = {m: {} for m in ("PatchTST", "TimesNet", "TimeMixer")}
    for r in rows:
        if r.get("dataset") != "ETTh1" or r.get("input_len") != 96:
            continue
        m, lab = r["model"], r["wide_label"]
        if m in log_map and lab in ("2xW", "2xD", "2xB"):
            log_map[m][lab] = r["nparams"]
    for m in ("PatchTST", "TimesNet", "TimeMixer"):
        raw = stats[m]["raw_params"]
        extra = f"  {m}: Raw 1.0$\\times$"
        for lab in ("2xW", "2xD", "2xB"):
            if m in log_map and lab in log_map[m]:
                extra += f"  |  {lab} {_fmt_ratio(log_map[m][lab], raw)}"
        dpr = stats[m]["dpr_params"]
        extra += f"  |  +DPR {_fmt_ratio(dpr, raw)}"
        print(extra)
    print()
    print("Note: FLOPs are not in .log files. --flops uses thop (slow on large TimesNet).")
    print("      The ~4.0$\\times$ / ~2.0$\\times$ / ~8.0$\\times$ in drafts are design targets;")
    print("      use the line above for measured param ratios from your checkpoints.")


if __name__ == "__main__":
    main()
