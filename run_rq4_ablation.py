#!/usr/bin/env python3
"""
RQ2 Ablation Study: DPR Component Analysis on Illness and ExchangeRate.

Ablation variants:
    1) w/o Multi-scale  : use_multiscale=False (k=1 conv instead of k1=3, k2=7)
    2) w/o Orthogonal   : orth_lambda=0 (no orthogonality regularization)
    3) w/o Identity Init: identity_init=False (gamma ~ N(0, 0.01) instead of 0)
    4) Discrete Top-2   : discrete_topk=2 (hard Top-2 routing instead of soft)

Models: PatchTST, TimeMixer, Informer, Crossformer
Datasets: Illness (7 feat, 24->24), ExchangeRate (8 feat, 96->96)
"""
import os
import sys
import time
from multiprocessing import Process, Queue

script_dir = os.path.dirname(os.path.abspath(__file__))
RQ2_CKPT_SAVE_DIR = os.path.join(script_dir, "checkpoints", "test_ablation")
src_dir = os.path.join(script_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from basicts.models.PatchTST import PatchTSTForForecasting, PatchTSTConfig
from basicts.models.TimeMixer import TimeMixerForForecasting, TimeMixerConfig
from basicts.models.Informer import Informer, InformerConfig
from basicts.models.Crossformer import Crossformer, CrossformerConfig
from basicts.configs import BasicTSForecastingConfig, DPRConfig
from basicts.runners.callback import EarlyStopping
from basicts import BasicTSLauncher

AVAILABLE_GPUS = [0, 1, 2, 3, 4, 5, 6, 7]
JOBS_PER_GPU = 4

RQ2_TASKS = [
    # PatchTST & TimeMixer
    ("PatchTST",    "Illness",      7, 24, 24),
    ("TimeMixer",   "Illness",      7, 24, 24),
    ("PatchTST",    "ExchangeRate",  8, 96, 96),
    ("TimeMixer",   "ExchangeRate",  8, 96, 96),
    # Informer (needs label_len parameter)
    ("Informer",    "Illness",      7, 24, 24),
    ("Informer",    "ExchangeRate",  8, 96, 96),
    # Crossformer (needs patch_len that divides input_len; 24/8=3, 96/16=6)
    ("Crossformer", "Illness",      7, 24, 24),
    ("Crossformer", "ExchangeRate",  8, 96, 96),
]

ABLATION_CONFIGS = [
    {"tag": "wo_multi_scale",  "use_multiscale": False, "identity_init": True,  "discrete_topk": 1, "orth_lambda": 0.01},
    {"tag": "wo_orth",         "use_multiscale": True,  "identity_init": True,  "discrete_topk": 1, "orth_lambda": 0.0},
    {"tag": "wo_identity",     "use_multiscale": True,  "identity_init": False, "discrete_topk": 1, "orth_lambda": 0.01},
    {"tag": "discrete_top2",   "use_multiscale": True,  "identity_init": True,  "discrete_topk": 2, "orth_lambda": 0.01},
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


def get_model_config(model_name, input_len, output_len, num_features, dataset_name, overrides=None):
    overrides = dict(overrides or {})
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
    if model_name == "Informer":
        cfg = InformerConfig(
            input_len=input_len,
            output_len=output_len,
            label_len=input_len,
            num_features=num_features,
        )
        _apply_overrides(cfg, overrides)
        return Informer, cfg, False
    if model_name == "Crossformer":
        patch_len = 8 if input_len == 24 else 16
        cfg = CrossformerConfig(
            input_len=input_len,
            output_len=output_len,
            num_features=num_features,
            patch_len=patch_len,
        )
        _apply_overrides(cfg, overrides)
        return Crossformer, cfg, False
    raise ValueError(f"Unknown model: {model_name}")


def run_experiment(
    model_name,
    dataset_name,
    num_features,
    input_len,
    output_len,
    gpu_id,
    model_overrides=None,
    dpr_overrides=None,
):
    model_class, model_config, use_timestamps = get_model_config(
        model_name, input_len, output_len, num_features, dataset_name, overrides=model_overrides
    )

    callbacks = [EarlyStopping(patience=10)]

    dpr_cfg = DPRConfig(
        enabled=True,
        num_patterns=8,
        orth_lambda=dpr_overrides.get("orth_lambda", 0.01) if dpr_overrides else 0.01,
        use_multiscale=dpr_overrides.get("use_multiscale", True) if dpr_overrides else True,
        identity_init=dpr_overrides.get("identity_init", True) if dpr_overrides else True,
        discrete_topk=dpr_overrides.get("discrete_topk", 1) if dpr_overrides else 1,
    )
    if hasattr(model_config, "dpr"):
        model_config.dpr = dpr_cfg

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
    dpr_overrides,
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
            dpr_overrides=dpr_overrides,
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
    gpu_queue = Queue()
    for gid in AVAILABLE_GPUS:
        for _ in range(JOBS_PER_GPU):
            gpu_queue.put(gid)

    processes = []
    print(
        f"RQ2 Ablation study on GPUs {AVAILABLE_GPUS} "
        f"(≤{len(AVAILABLE_GPUS) * JOBS_PER_GPU} concurrent)"
    )
    print(f"Tasks: {len(RQ2_TASKS)} models × {len(ABLATION_CONFIGS)} ablations = {len(RQ2_TASKS) * len(ABLATION_CONFIGS)} total\n")

    for model_name, dataset_name, num_features, input_len, output_len in RQ2_TASKS:
        for abl_cfg in ABLATION_CONFIGS:
            tag = f"RQ2_ABL_{abl_cfg['tag']}"
            dpr_overrides = {
                "orth_lambda": abl_cfg["orth_lambda"],
                "use_multiscale": abl_cfg["use_multiscale"],
                "identity_init": abl_cfg["identity_init"],
                "discrete_topk": abl_cfg["discrete_topk"],
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
                    None,
                    dpr_overrides,
                ),
            )
            p.start()
            processes.append(p)
            time.sleep(0.05)

    print(f"Scheduled {len(processes)} ablation experiments.")

    for p in processes:
        p.join()

    print("All RQ2 ablation experiments finished.")
