#!/usr/bin/env python3
"""
RQ2 Wide Scaling Sweep: PatchTST / TimesNet / TimeMixer, or WPMixer / TimeFilter, on
  - ETTh1       : input 96 -> pred 96
  - Illness     : input 24 -> pred 24
  - ExchangeRate: input 96 -> pred 96

Only runs WIDE scaling variants (no RAW / DPR — results already in dpr_result.md).
Each (model, dataset) runs 4 scaling configs:
  1) 2x width     : hidden=512,  intermediate=2048, num_layers=1
  2) 2x depth     : hidden=256,  intermediate=1024, num_layers=2
  3) 2x both      : hidden=512,  intermediate=2048, num_layers=2
  4) param_match  : width auto-tuned so base model params ≈ base+DPR params

Outputs (see `RQ2_CKPT_SAVE_DIR`): each job saves under ``{RQ2_CKPT_SAVE_DIR}/{md5}/`` —
checkpoints (``*.pt``), ``training_log_*.log``, ``tensorboard/``, ``cfg.json``, ``test_metrics.json``.
"""
import os
import sys
import time
from multiprocessing import Process, Queue

import torch

script_dir = os.path.dirname(os.path.abspath(__file__))
RQ2_CKPT_SAVE_DIR = os.path.join(script_dir, "checkpoints", "test_scaling_match")
src_dir = os.path.join(script_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from basicts.models.PatchTST import PatchTSTForForecasting, PatchTSTConfig
from basicts.models.TimeFilter import TimeFilterForForecasting, TimeFilterConfig
from basicts.models.TimeMixer import TimeMixerForForecasting, TimeMixerConfig
from basicts.models.TimesNet import TimesNetForForecasting, TimesNetConfig
from basicts.models.WPMixer import WPMixerForForecasting, WPMixerConfig
from basicts.configs import BasicTSForecastingConfig, DPRConfig
from basicts.runners.callback import EarlyStopping
from basicts import BasicTSLauncher

AVAILABLE_GPUS = [0, 1, 3, 4, 5, 6, 7]
JOBS_PER_GPU = 2

# (model_name, dataset_name, num_features, input_len, output_len)
RQ2_TASKS = [
    ("PatchTST",    "ETTh1",        7, 96, 96),
    ("TimesNet",    "ETTh1",        7, 96, 96),
    # ("TimeMixer",   "ETTh1",        7, 96, 96),
    # ("WPMixer",     "ETTh1",        7, 96, 96),
    ("TimeFilter",  "ETTh1",        7, 96, 96),
    ("PatchTST",    "Illness",      7, 24, 24),
    ("TimesNet",    "Illness",      7, 24, 24),
    # ("TimeMixer",   "Illness",      7, 24, 24),
    # ("WPMixer",     "Illness",      7, 24, 24),
    ("TimeFilter",  "Illness",      7, 24, 24),
    ("PatchTST",    "ExchangeRate", 8, 96, 96),
    ("TimesNet",    "ExchangeRate", 8, 96, 96),
    # ("TimeMixer",   "ExchangeRate", 8, 96, 96),
    # ("WPMixer",     "ExchangeRate", 8, 96, 96),
    ("TimeFilter",  "ExchangeRate", 8, 96, 96),
]

# 3 representative WIDE configs (all ~2x scaling factor vs base):
#   - 2x width : only width scaled
#   - 2x depth : only depth scaled
#   - 2x both  : width and depth both scaled
WIDE_CONFIGS = [
    # {"tag": "w2_d1", "hidden_size": 512,  "intermediate_size": 2048, "num_layers": 1},  # 2x width
    # {"tag": "w1_d2", "hidden_size": 256,  "intermediate_size": 1024, "num_layers": 2},  # 2x depth
    # {"tag": "w2_d2", "hidden_size": 512,  "intermediate_size": 2048, "num_layers": 2},  # 2x both
    {"tag": "param_match", "hidden_size": 0, "intermediate_size": 0, "num_layers": 1},  # params ≈ base+DPR
]

USE_CLEAN_TARGETS = True


def get_timestamp_sizes(dataset_name: str):
    if dataset_name == "ExchangeRate":
        return [1, 7, 31, 366]
    if dataset_name == "Traffic":
        return [24, 7, 31, 366]
    if dataset_name in ["ETTh1", "ETTh2"]:
        return [24, 7, 31, 366]
    if dataset_name in ["ETTm1", "ETTm2", "SyntheticTS"]:
        return [96, 7, 31, 366]
    return [60, 7, 31, 366]


def _apply_overrides(cfg, overrides: dict | None):
    if not overrides:
        return
    for k, v in overrides.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
        else:
            raise ValueError(f"Unknown field {k} on {type(cfg).__name__}")


def _count_params(model_name, dataset_name, num_features, input_len, output_len, overrides, use_dpr):
    mclass, cfg, ts = get_model_config(
        model_name, input_len, output_len, num_features, dataset_name, overrides=overrides
    )
    if hasattr(cfg, "dpr"):
        cfg.dpr = DPRConfig(enabled=use_dpr)
    model = mclass(cfg)
    model = model.cpu()
    n = sum(p.numel() for p in model.parameters())
    del model
    return n


def find_param_matched_overrides(model_name, dataset_name, num_features, input_len, output_len):
    target = _count_params(model_name, dataset_name, num_features,
                           input_len, output_len, None, use_dpr=True)

    candidates = [32, 48, 64, 96, 128, 160, 192, 224, 256, 288, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024]
    best_diff = float("inf")
    best_overrides = None
    best_params = 0

    for h in candidates:
        ff = max(64, h * 4)
        overrides = {"hidden_size": h, "intermediate_size": ff, "num_layers": 1}
        try:
            params = _count_params(model_name, dataset_name, num_features,
                                   input_len, output_len, overrides, use_dpr=False)
            diff = abs(params - target)
            if diff < best_diff:
                best_diff = diff
                best_params = params
                best_overrides = dict(overrides)
        except Exception:
            continue

    if best_overrides is None:
        return None
    if best_diff > target * 0.5:
        print(f"  [param_match] {model_name}/{dataset_name}: "
              f"target={target / 1e6:.2f}M best={best_params / 1e6:.2f}M "
              f"(diff={best_diff / 1e6:.2f}M > 50% of target, skipping)")
        return None
    print(f"  [param_match] {model_name}/{dataset_name}: "
          f"target={target / 1e6:.2f}M -> matched={best_params / 1e6:.2f}M, "
          f"hidden={best_overrides['hidden_size']}")
    return best_overrides


def get_model_config(model_name, input_len, output_len, num_features, dataset_name, overrides=None):
    o = dict(overrides or {})
    if model_name == "WPMixer" and o:
        h, ff, _ = o["hidden_size"], o["intermediate_size"], o["num_layers"]
        exp = max(2, min(20, ff // max(h, 1)))
        o = {"d_model": h, "tfactor": exp, "dfactor": exp}
    elif model_name == "TimeFilter" and o:
        h, ff, nl = o["hidden_size"], o["intermediate_size"], o["num_layers"]
        o = {"d_model": h, "d_ff": ff, "e_layers": nl}
    overrides = o
    if model_name == "PatchTST":
        cfg = PatchTSTConfig(
            input_len=input_len,
            output_len=output_len,
            num_features=num_features,
        )
        _apply_overrides(cfg, overrides)
        return PatchTSTForForecasting, cfg, False
    if model_name == "TimeMixer":
        cfg = TimeMixerConfig(
            input_len=input_len,
            output_len=output_len,
            num_features=num_features,
        )
        _apply_overrides(cfg, overrides)
        return TimeMixerForForecasting, cfg, False
    if model_name == "TimesNet":
        cfg = TimesNetConfig(
            input_len=input_len,
            output_len=output_len,
            num_features=num_features,
            use_timestamps=True,
            timestamp_sizes=get_timestamp_sizes(dataset_name),
        )
        _apply_overrides(cfg, overrides)
        return TimesNetForForecasting, cfg, True
    if model_name == "WPMixer":
        cfg = WPMixerConfig(
            input_len=input_len,
            output_len=output_len,
            num_features=num_features,
        )
        _apply_overrides(cfg, overrides)
        return WPMixerForForecasting, cfg, False
    if model_name == "TimeFilter":
        cfg = TimeFilterConfig(
            input_len=input_len,
            output_len=output_len,
            num_features=num_features,
        )
        _apply_overrides(cfg, overrides)
        return TimeFilterForForecasting, cfg, False
    raise ValueError(f"Unknown model: {model_name}")


def run_experiment(
    model_name,
    dataset_name,
    num_features,
    input_len,
    output_len,
    gpu_id,
    model_overrides=None,
):
    model_class, model_config, use_timestamps = get_model_config(
        model_name, input_len, output_len, num_features, dataset_name, overrides=model_overrides
    )

    callbacks = [EarlyStopping(patience=10)]
    # DPR disabled for WIDE runs
    if hasattr(model_config, "dpr"):
        model_config.dpr = DPRConfig(enabled=False)

    cfg = BasicTSForecastingConfig(
        model=model_class,
        model_config=model_config,
        dataset_name=dataset_name,
        input_len=input_len,
        output_len=output_len,
        use_timestamps=use_timestamps,
        use_clean_targets=USE_CLEAN_TARGETS,
        gpus=gpu_id,
        ckpt_save_dir=RQ2_CKPT_SAVE_DIR,
        num_epochs=100,
        batch_size=64,
        callbacks=callbacks,
        seed=42,
        train_data_num_workers=16,
        val_data_num_workers=16,
        test_data_num_workers=16,
        train_data_pin_memory=True,
        val_data_pin_memory=True,
        test_data_pin_memory=True,
    )
    BasicTSLauncher.launch_training(cfg)


def worker(
    gpu_queue,
    model_name,
    dataset_name,
    num_features,
    input_len,
    output_len,
    tag: str,
    model_overrides,
):
    gpu_id = None
    task_id = f"{tag} | {model_name} | {dataset_name} | {input_len}->{output_len}"
    try:
        gpu_id = gpu_queue.get()
        print(f"[Start] {task_id} on GPU {gpu_id}")
        run_experiment(
            model_name,
            dataset_name,
            num_features,
            input_len,
            output_len,
            str(gpu_id),
            model_overrides=model_overrides,
        )
    except Exception as e:
        print(f"[Error] {task_id} failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if gpu_id is not None:
            gpu_queue.put(gpu_id)
            print(f"[Done] {task_id} released GPU {gpu_id}")


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    gpu_queue = Queue()
    for gid in AVAILABLE_GPUS:
        for _ in range(JOBS_PER_GPU):
            gpu_queue.put(gid)

    processes = []
    print(
        f"RQ2 WIDE scaling sweep on GPUs {AVAILABLE_GPUS} "
        f"(≤{len(AVAILABLE_GPUS) * JOBS_PER_GPU} concurrent)"
    )
    print(f"Tasks: {len(RQ2_TASKS)} models × {len(WIDE_CONFIGS)} WIDE configs = {len(RQ2_TASKS) * len(WIDE_CONFIGS)} total\n")

    for model_name, dataset_name, num_features, input_len, output_len in RQ2_TASKS:
        for wide_cfg in WIDE_CONFIGS:
            tag = f"RQ2_WIDE_{wide_cfg['tag']}"
            if wide_cfg["tag"] == "param_match":
                overrides = find_param_matched_overrides(
                    model_name, dataset_name, num_features, input_len, output_len
                )
                if overrides is None:
                    print(f"  [skip] param_match failed for {model_name}/{dataset_name}")
                    continue
            else:
                overrides = {
                    "hidden_size": wide_cfg["hidden_size"],
                    "intermediate_size": wide_cfg["intermediate_size"],
                    "num_layers": wide_cfg["num_layers"],
                }
            p = Process(
                target=worker,
                args=(
                    gpu_queue,
                    model_name,
                    dataset_name,
                    num_features,
                    input_len,
                    output_len,
                    tag,
                    overrides,
                ),
            )
            p.start()
            processes.append(p)
            time.sleep(0.05)

    print(f"Scheduled {len(processes)} WIDE experiments.")

    for p in processes:
        p.join()

    print("All RQ2 WIDE experiments finished.")