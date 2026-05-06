#!/usr/bin/env python3
"""
Sensitivity Analysis for RQ5: PatchTST + Crossformer on Illness dataset

Three sensitivity curves:
  Curve 1: K=8 fixed, orth_lambda varies in {0, 0.0001, 0.001, 0.01, 0.1}
  Curve 2: orth_lambda=0.0001 fixed, K varies in {4, 8, 16, 32}
  Curve 3: K=8 and orth_lambda=0.0001 fixed, conv kernels vary in
           {(1,), (3,), (5,), (7,), (3, 7)}

Outputs saved under checkpoints/test_sensitivity/{md5}/
"""
import os
import sys
import time
import argparse
from multiprocessing import Process, Queue

script_dir = os.path.dirname(os.path.abspath(__file__))
SENSITIVITY_CKPT_SAVE_DIR = os.path.join(script_dir, "checkpoints", "test_sensitivity_conv")
src_dir = os.path.join(script_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from basicts.models.PatchTST import PatchTSTForForecasting, PatchTSTConfig
from basicts.models.Crossformer import Crossformer, CrossformerConfig
from basicts.configs import BasicTSForecastingConfig, DPRConfig
from basicts.runners.callback import AddAuxiliaryLoss, EarlyStopping
from basicts import BasicTSLauncher

AVAILABLE_GPUS = [0, 1, 2, 3, 4, 5, 6, 7]
JOBS_PER_GPU = 4

MODEL_NAMES = ["PatchTST", "Crossformer"]
DATASET_NAME = "Illness"
NUM_FEATURES = 7
INPUT_LEN = 24
OUTPUT_LENS = [24, 36, 48, 60]

ORTH_LAMBDA_SENSITIVITY = [0.0, 0.0001, 0.001, 0.01, 0.1]
K_SENSITIVITY = [4, 8, 16, 32]
CONV_KERNEL_SENSITIVITY = [(1,), (3,), (5,), (7,), (3, 7)]
K_FIXED = 8
ORTH_LAMBDA_FIXED = 0.0001
CONV_KERNELS_FIXED = (3, 7)

USE_CLEAN_TARGETS = False


def _format_orth_lambda(x: float) -> str:
    if x == 0.0:
        return "0"
    exp = f"{x:.0e}"
    return exp if "e" in exp.lower() else f"{x:g}"


def _format_conv_kernels(conv_kernels: tuple[int, ...]) -> str:
    return "+".join(str(k) for k in conv_kernels)


def get_model_config(
    model_name,
    input_len,
    output_len,
    num_features,
    dpr_num_patterns,
    dpr_orth_lambda,
    dpr_conv_kernels,
):
    if model_name == "PatchTST":
        cfg = PatchTSTConfig(
            input_len=input_len,
            output_len=output_len,
            num_features=num_features,
        )
        model_class = PatchTSTForForecasting
    elif model_name == "Crossformer":
        cfg = CrossformerConfig(
            input_len=input_len,
            output_len=output_len,
            num_features=num_features,
            patch_len=8,
        )
        model_class = Crossformer
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    cfg.dpr = DPRConfig(
        enabled=True,
        num_patterns=int(dpr_num_patterns),
        orth_lambda=float(dpr_orth_lambda),
        conv_kernels=tuple(int(k) for k in dpr_conv_kernels),
    )
    return model_class, cfg, False


def run_experiment(
    model_name,
    input_len,
    output_len,
    gpu_id,
    dpr_num_patterns,
    dpr_orth_lambda,
    dpr_conv_kernels,
):
    model_class, model_config, use_timestamps = get_model_config(
        model_name,
        input_len,
        output_len,
        NUM_FEATURES,
        dpr_num_patterns,
        dpr_orth_lambda,
        dpr_conv_kernels,
    )

    callbacks = [EarlyStopping(patience=10)]
    if dpr_orth_lambda > 0:
        callbacks.append(AddAuxiliaryLoss(losses=["dpr_orth"]))

    cfg = BasicTSForecastingConfig(
        model=model_class,
        model_config=model_config,
        dataset_name=DATASET_NAME,
        input_len=input_len,
        output_len=output_len,
        use_timestamps=use_timestamps,
        use_clean_targets=USE_CLEAN_TARGETS,
        gpus=gpu_id,
        ckpt_save_dir=SENSITIVITY_CKPT_SAVE_DIR,
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
    input_len,
    output_len,
    dpr_num_patterns,
    dpr_orth_lambda,
    dpr_conv_kernels,
):
    gpu_id = None
    task_id = (
        f"{model_name} | {DATASET_NAME} | {input_len}->{output_len} | "
        f"K={dpr_num_patterns} orth={_format_orth_lambda(dpr_orth_lambda)} "
        f"conv={_format_conv_kernels(dpr_conv_kernels)}"
    )
    try:
        gpu_id = gpu_queue.get()
        print(f"[Start] {task_id} on GPU {gpu_id}")
        run_experiment(
            model_name,
            input_len,
            output_len,
            str(gpu_id),
            dpr_num_patterns,
            dpr_orth_lambda,
            dpr_conv_kernels,
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
    parser = argparse.ArgumentParser(description="Run RQ5 sensitivity experiments")
    parser.add_argument(
        "--only-conv",
        action="store_true",
        help="Run only Curve 3 (conv kernel sensitivity)",
    )
    args = parser.parse_args()

    gpu_queue = Queue()
    for gid in AVAILABLE_GPUS:
        for _ in range(JOBS_PER_GPU):
            gpu_queue.put(gid)

    processes = []
    print(f"=== Sensitivity Analysis ===")
    print(f"Models: {MODEL_NAMES}, Dataset: {DATASET_NAME}")
    print(f"GPU: {AVAILABLE_GPUS}, Concurrent: {len(AVAILABLE_GPUS) * JOBS_PER_GPU}")
    print()
    print(f"Curve 1: K={K_FIXED} fixed, orth_lambda varies:")
    print(f"  orth_lambda: {ORTH_LAMBDA_SENSITIVITY}")
    print()
    print(f"Curve 2: orth_lambda={ORTH_LAMBDA_FIXED} fixed, K varies:")
    print(f"  K: {K_SENSITIVITY}")
    print()
    print(f"Curve 3: K={K_FIXED}, orth_lambda={ORTH_LAMBDA_FIXED} fixed, conv kernels vary:")
    print(f"  conv_kernels: {CONV_KERNEL_SENSITIVITY}")
    print()

    total = 0

    if not args.only_conv:
        for model_name in MODEL_NAMES:
            for output_len in OUTPUT_LENS:
                for orth_lambda in ORTH_LAMBDA_SENSITIVITY:
                    p = Process(
                        target=worker,
                        args=(
                            gpu_queue,
                            model_name,
                            INPUT_LEN,
                            output_len,
                            K_FIXED,
                            orth_lambda,
                            CONV_KERNELS_FIXED,
                        ),
                    )
                    p.start()
                    processes.append(p)
                    time.sleep(0.1)
                    total += 1

        for model_name in MODEL_NAMES:
            for output_len in OUTPUT_LENS:
                for k in K_SENSITIVITY:
                    if k == K_FIXED:
                        continue
                    p = Process(
                        target=worker,
                        args=(
                            gpu_queue,
                            model_name,
                            INPUT_LEN,
                            output_len,
                            k,
                            ORTH_LAMBDA_FIXED,
                            CONV_KERNELS_FIXED,
                        ),
                    )
                    p.start()
                    processes.append(p)
                    time.sleep(0.1)
                    total += 1

    for model_name in MODEL_NAMES:
        for output_len in OUTPUT_LENS:
            for conv_kernels in CONV_KERNEL_SENSITIVITY:
                if conv_kernels == CONV_KERNELS_FIXED:
                    continue
                p = Process(
                    target=worker,
                    args=(
                        gpu_queue,
                        model_name,
                        INPUT_LEN,
                        output_len,
                        K_FIXED,
                        ORTH_LAMBDA_FIXED,
                        conv_kernels,
                    ),
                )
                p.start()
                processes.append(p)
                time.sleep(0.1)
                total += 1

    print(f"Scheduled {total} experiments")
    print(f"Checkpoints saved to: {SENSITIVITY_CKPT_SAVE_DIR}")

    for p in processes:
        p.join()

    print("All sensitivity experiments finished.")
