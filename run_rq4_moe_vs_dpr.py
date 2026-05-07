#!/usr/bin/env python3
"""
MoE vs DPR Comparison: Compare DPRNet and MoE-DPRNet on selected datasets.

Datasets:
    - Illness (7 feat, 24->24/36/48/60)
    - ExchangeRate (8 feat, 96->96/192/336/720)
    - ETTh1 (7 feat, 96->96/192/336/720)

Models:
    - DPRNet: Original with DPR
    - MoE-DPRNet: MoE replacement for DPR
"""
import os
import sys
import time
from multiprocessing import Process, Queue

script_dir = os.path.dirname(os.path.abspath(__file__))
CKPT_SAVE_DIR = os.path.join(script_dir, "checkpoints", "moe_vs_dpr")
src_dir = os.path.join(script_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from basicts.models.DPRNet import DPRNetForForecasting, DPRNetConfig
from basicts.models.MoEDPRNet import MoEDPRNetForForecasting, MoEDPRNetConfig
from basicts.configs import BasicTSForecastingConfig
from basicts.runners.callback import EarlyStopping, AddAuxiliaryLoss
from basicts import BasicTSLauncher

AVAILABLE_GPUS = [0, 1, 2, 3, 4, 5, 6, 7]
JOBS_PER_GPU = 1

DATASET_CONFIGS = {
    "Illness": {
        "num_features": 7,
        "input_len": 24,
        "output_lens": [24],
    },
    "ExchangeRate": {
        "num_features": 8,
        "input_len": 96,
        "output_lens": [96],
    },
    "ETTh1": {
        "num_features": 7,
        "input_len": 96,
        "output_lens": [96],
    },
}


def run_experiment(
    model_name,
    dataset_name,
    num_features,
    input_len,
    output_len,
    gpu_id,
    top_k=None,
):
    if model_name == "DPRNet":
        cfg = DPRNetConfig(
            input_len=input_len,
            output_len=output_len,
            num_features=num_features,
            patch_len=16,
            patch_stride=8,
            hidden_size=256,
            num_mlp_layers=2,
            mlp_expansion=2.0,
            mlp_dropout=0.1,
            use_dpr=True,
            num_patterns=8,
            use_multiscale=True,
            identity_init=True,
            orth_lambda=0.01,
            head_dropout=0.0,
            use_revin=True,
        )
        model_class = DPRNetForForecasting
        callbacks = [EarlyStopping(patience=10)]
        if cfg.orth_lambda > 0:
            callbacks.append(AddAuxiliaryLoss(losses=["dpr_orth"]))
    elif model_name == "DPRNet_no_DPR":
        cfg = DPRNetConfig(
            input_len=input_len,
            output_len=output_len,
            num_features=num_features,
            patch_len=16,
            patch_stride=8,
            hidden_size=256,
            num_mlp_layers=2,
            mlp_expansion=2.0,
            mlp_dropout=0.1,
            use_dpr=False,
            num_patterns=8,
            use_multiscale=True,
            identity_init=True,
            orth_lambda=0.0,
            head_dropout=0.0,
            use_revin=True,
        )
        model_class = DPRNetForForecasting
        callbacks = [EarlyStopping(patience=10)]
    elif model_name.startswith("MoEDPRNet"):
        top_k_value = top_k if top_k is not None else 1
        cfg = MoEDPRNetConfig(
            input_len=input_len,
            output_len=output_len,
            num_features=num_features,
            patch_len=16,
            patch_stride=8,
            hidden_size=256,
            num_mlp_layers=2,
            mlp_expansion=2.0,
            mlp_dropout=0.1,
            num_experts=8,
            top_k=top_k_value,
            noisy_gating=True,
            moe_loss_coef=0.01,
            head_dropout=0.0,
            use_revin=True,
        )
        model_class = MoEDPRNetForForecasting
        callbacks = [EarlyStopping(patience=10)]
        if cfg.moe_loss_coef > 0:
            callbacks.append(AddAuxiliaryLoss(losses=["moe_loss"]))
    else:
        raise ValueError(f"Unknown model: {model_name}")

    basic_cfg = BasicTSForecastingConfig(
        model=model_class,
        model_config=cfg,
        dataset_name=dataset_name,
        input_len=input_len,
        output_len=output_len,
        use_timestamps=False,
        use_clean_targets=False,
        gpus=gpu_id,
        ckpt_save_dir=CKPT_SAVE_DIR,
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
    BasicTSLauncher.launch_training(basic_cfg)


def worker(
    gpu_queue,
    model_name,
    dataset_name,
    num_features,
    input_len,
    output_len,
    top_k=None,
):
    gpu_id = None
    top_k_str = f"_top{top_k}" if top_k is not None else ""
    task_id = f"{model_name}{top_k_str} | {dataset_name} | {input_len}->{output_len}"
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
            top_k=top_k,
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
        f"MoE vs DPR comparison on GPUs {AVAILABLE_GPUS} "
        f"(up to {len(AVAILABLE_GPUS) * JOBS_PER_GPU} concurrent)"
    )

    total_tasks = 0
    for dataset_name, ds_config in DATASET_CONFIGS.items():
        num_features = ds_config["num_features"]
        input_len = ds_config["input_len"]
        for output_len in ds_config["output_lens"]:
            p = Process(
                target=worker,
                args=(
                    gpu_queue,
                    "DPRNet_no_DPR",
                    dataset_name,
                    num_features,
                    input_len,
                    output_len,
                    None,
                ),
            )
            p.start()
            processes.append(p)
            time.sleep(0.05)
            total_tasks += 1
            
            for top_k in [1, 2, 4]:
                p = Process(
                    target=worker,
                    args=(
                        gpu_queue,
                        "MoEDPRNet",
                        dataset_name,
                        num_features,
                        input_len,
                        output_len,
                        top_k,
                    ),
                )
                p.start()
                processes.append(p)
                time.sleep(0.05)
                total_tasks += 1

    print(f"Scheduled {total_tasks} comparison experiments.")
    print(f"Datasets: {list(DATASET_CONFIGS.keys())}")
    print(f"Models: DPRNet_no_DPR, MoEDPRNet (top_k=1,2,4)")

    for p in processes:
        p.join()

    print("All MoE vs DPR comparison experiments finished.")
