import sys
import os
import time
from multiprocessing import Process, Queue

script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from basicts.models.Crossformer import Crossformer, CrossformerConfig
from basicts.models.Informer import Informer, InformerConfig
from basicts.models.iTransformer import iTransformerForForecasting, iTransformerConfig
from basicts.models.PatchTST import PatchTSTForForecasting, PatchTSTConfig
from basicts.models.TimeMixer import TimeMixerForForecasting, TimeMixerConfig
from basicts.models.TimesNet import TimesNetForForecasting, TimesNetConfig
from basicts.configs import BasicTSForecastingConfig, TCGConfig
from basicts.runners.callback import AddAuxiliaryLoss, EarlyStopping
from basicts import BasicTSLauncher

AVAILABLE_GPUS = [0, 1, 2, 3, 4, 5, 6, 7]
# Each GPU id is queued this many times → max concurrent runs = len(AVAILABLE_GPUS) * JOBS_PER_GPU.
# 2 → 16 workers on 8 GPUs (shared cards; reduce batch_size or set 1 if OOM).
JOBS_PER_GPU = 6

MISSING_TCG_EXPERIMENTS = [
    # ETTh1
    ("PatchTST", "ETTh1", 7, 96, 96),
    ("PatchTST", "ETTh1", 7, 96, 720),
    ("TimesNet", "ETTh1", 7, 96, 192),
    ("TimesNet", "ETTh1", 7, 96, 720),
    ("TimeMixer", "ETTh1", 7, 96, 192),
    ("TimeMixer", "ETTh1", 7, 96, 336),
    ("Crossformer", "ETTh1", 7, 96, 720),
    # ETTh2
    ("PatchTST", "ETTh2", 7, 96, 96),
    ("PatchTST", "ETTh2", 7, 96, 192),
    ("TimeMixer", "ETTh2", 7, 96, 96),
    ("TimeMixer", "ETTh2", 7, 96, 192),
    ("TimeMixer", "ETTh2", 7, 96, 336),
    ("Crossformer", "ETTh2", 7, 96, 336),
    ("TimesNet", "ETTh2", 7, 96, 336),
    ("TimesNet", "ETTh2", 7, 96, 720),
    ("Informer", "ETTh2", 7, 96, 720),
    # ETTm1
    ("PatchTST", "ETTm1", 7, 96, 96),
    ("PatchTST", "ETTm1", 7, 96, 336),
    ("TimesNet", "ETTm1", 7, 96, 96),
    ("TimesNet", "ETTm1", 7, 96, 192),
    ("TimesNet", "ETTm1", 7, 96, 720),
    ("TimeMixer", "ETTm1", 7, 96, 96),
    ("TimeMixer", "ETTm1", 7, 96, 192),
    ("Crossformer", "ETTm1", 7, 96, 336),
    # ETTm2
    ("PatchTST", "ETTm2", 7, 96, 96),
    ("PatchTST", "ETTm2", 7, 96, 192),
    ("PatchTST", "ETTm2", 7, 96, 336),
    ("PatchTST", "ETTm2", 7, 96, 720),
    ("TimeMixer", "ETTm2", 7, 96, 192),
    ("TimeMixer", "ETTm2", 7, 96, 336),
    ("TimeMixer", "ETTm2", 7, 96, 720),
    ("Crossformer", "ETTm2", 7, 96, 96),
    ("Crossformer", "ETTm2", 7, 96, 192),
    ("Crossformer", "ETTm2", 7, 96, 336),
    ("TimesNet", "ETTm2", 7, 96, 96),
    ("TimesNet", "ETTm2", 7, 96, 192),
    ("TimesNet", "ETTm2", 7, 96, 336),
    ("Informer", "ETTm2", 7, 96, 336),
    ("Informer", "ETTm2", 7, 96, 720),
    # Weather
    ("Crossformer", "Weather", 21, 96, 96),
    ("TimesNet", "Weather", 21, 96, 96),
    ("TimesNet", "Weather", 21, 96, 336),
    ("Informer", "Weather", 21, 96, 336),
    ("PatchTST", "Weather", 21, 96, 336),
    ("PatchTST", "Weather", 21, 96, 720),
    ("Crossformer", "Weather", 21, 96, 720),
    # ECL
    ("Informer", "ECL", 321, 96, 96),
    ("Informer", "ECL", 321, 96, 192),
    ("Informer", "ECL", 321, 96, 336),
    ("PatchTST", "ECL", 321, 96, 192),
    ("PatchTST", "ECL", 321, 96, 336),
    ("PatchTST", "ECL", 321, 96, 720),
    # ILI
    ("TimeMixer", "ILI", 1, 24, 24),
    ("TimeMixer", "ILI", 1, 24, 48),
    ("TimeMixer", "ILI", 1, 24, 60),
    ("Crossformer", "ILI", 1, 24, 60),
    ("Informer", "ILI", 1, 24, 60),
    ("PatchTST", "ILI", 1, 24, 60),
    # ExchangeRate
    ("Informer", "ExchangeRate", 8, 96, 96),
    ("TimeMixer", "ExchangeRate", 8, 96, 192),
    ("TimeMixer", "ExchangeRate", 8, 96, 336),
    ("TimeMixer", "ExchangeRate", 8, 96, 720),
]

MISSING_RAW_EXPERIMENTS = [
    # ("Crossformer", "Traffic", 862, 96, 96),
    # ("Crossformer", "Traffic", 862, 96, 192),
    # ("Crossformer", "Traffic", 862, 96, 336),
    # ("Crossformer", "Traffic", 862, 96, 720),
    # ("TimeMixer", "Traffic", 862, 96, 96),
    # ("TimeMixer", "Traffic", 862, 96, 192),
    # ("TimeMixer", "Traffic", 862, 96, 336),
]

TCG_ORTH_LAMBDA_SEARCH = [1e-4, 1e-3]
TCG_NUM_PATTERNS_SEARCH = [8]
USE_CLEAN_TARGETS = False


def _format_orth_lambda(x: float) -> str:
    if x == 0.0:
        return "0"
    exp = f"{x:.0e}"
    return exp if "e" in exp.lower() else f"{x:g}"


def get_timestamp_sizes(dataset_name: str):
    if dataset_name == "ExchangeRate":
        return [1, 7, 31, 366]
    if dataset_name == "Traffic":
        return [24, 7, 31, 366]
    if dataset_name in ["ETTh1", "ETTh2"]:
        return [24, 7, 31, 366]
    if dataset_name in ["ETTm1", "ETTm2", "SyntheticTS"]:
        return [96, 7, 31, 366]
    if dataset_name.startswith("SyntheticTS"):
        return [96, 7, 31, 366]
    return [60, 7, 31, 366]


def get_model_config(model_name, input_len, output_len, num_features, dataset_name):
    if model_name == "Crossformer":
        cfg = CrossformerConfig(
            input_len=input_len,
            output_len=output_len,
            num_features=num_features
        )
        return Crossformer, cfg, False
    elif model_name == "Informer":
        cfg = InformerConfig(
            input_len=input_len,
            output_len=output_len,
            label_len=output_len // 2,
            num_features=num_features,
            use_timestamps=True,
            timestamp_sizes=get_timestamp_sizes(dataset_name)
        )
        return Informer, cfg, True
    elif model_name == "iTransformer":
        cfg = iTransformerConfig(
            input_len=input_len,
            output_len=output_len,
            num_features=num_features
        )
        return iTransformerForForecasting, cfg, False
    elif model_name == "PatchTST":
        cfg = PatchTSTConfig(
            input_len=input_len,
            output_len=output_len,
            num_features=num_features
        )
        return PatchTSTForForecasting, cfg, False
    elif model_name == "TimeMixer":
        cfg = TimeMixerConfig(
            input_len=input_len,
            output_len=output_len,
            num_features=num_features
        )
        return TimeMixerForForecasting, cfg, False
    elif model_name == "TimesNet":
        cfg = TimesNetConfig(
            input_len=input_len,
            output_len=output_len,
            num_features=num_features,
            use_timestamps=True,
            timestamp_sizes=get_timestamp_sizes(dataset_name)
        )
        return TimesNetForForecasting, cfg, True
    else:
        raise ValueError(f"Unknown model: {model_name}")


def run_experiment(model_name, dataset_name, num_features, input_len, output_len, gpu_id, **kwargs):
    model_class, model_config, use_timestamps = get_model_config(
        model_name, input_len, output_len, num_features, dataset_name
    )

    callbacks = [EarlyStopping(patience=10)]
    if kwargs.get("enable_tcg", True) and hasattr(model_config, "tcg"):
        tcg_cfg = TCGConfig(
            enabled=True,
            num_patterns=int(kwargs.get("tcg_num_patterns", kwargs.get("tcg_K", 8))),
            orth_lambda=float(kwargs.get("tcg_orth_lambda", 0.01)),
        )
        model_config.tcg = tcg_cfg
        if tcg_cfg.orth_lambda > 0:
            callbacks.append(AddAuxiliaryLoss(losses=["tcg_orth"]))

    cfg = BasicTSForecastingConfig(
        model=model_class, model_config=model_config,
        dataset_name=dataset_name, input_len=input_len, output_len=output_len,
        use_timestamps=use_timestamps, use_clean_targets=USE_CLEAN_TARGETS,
        gpus=gpu_id, num_epochs=100, batch_size=64, callbacks=callbacks, seed=42,
        train_data_num_workers=16, val_data_num_workers=16, test_data_num_workers=16,
        train_data_pin_memory=True, val_data_pin_memory=True, test_data_pin_memory=True,
    )
    BasicTSLauncher.launch_training(cfg)


def worker_task_tcg(
    gpu_queue,
    model_name,
    dataset_name,
    num_features,
    input_len,
    output_len,
    tcg_num_patterns,
    tcg_orth_lambda,
):
    gpu_id = None
    task_id = (
        f"{model_name} | {dataset_name} | {input_len}->{output_len} | "
        f"TCG K={tcg_num_patterns} orth={_format_orth_lambda(tcg_orth_lambda)}"
    )

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
            enable_tcg=True,
            tcg_num_patterns=tcg_num_patterns,
            tcg_orth_lambda=tcg_orth_lambda,
        )

    except Exception as e:
        print(f"[Error] {task_id} failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if gpu_id is not None:
            gpu_queue.put(gpu_id)
            print(f"[Done] {task_id} released GPU {gpu_id}")


def worker_task_raw(
    gpu_queue,
    model_name,
    dataset_name,
    num_features,
    input_len,
    output_len,
):
    gpu_id = None
    task_id = f"{model_name} | {dataset_name} | {input_len}->{output_len} | RAW"

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
            enable_tcg=False,
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
    for gpu_id in AVAILABLE_GPUS:
        for _ in range(JOBS_PER_GPU):
            gpu_queue.put(gpu_id)

    processes = []
    max_concurrent = len(AVAILABLE_GPUS) * JOBS_PER_GPU
    print(
        f"Scheduling tasks on GPUs {AVAILABLE_GPUS} "
        f"(up to {max_concurrent} concurrent, JOBS_PER_GPU={JOBS_PER_GPU})"
    )

    tcg_combos = len(TCG_NUM_PATTERNS_SEARCH) * len(TCG_ORTH_LAMBDA_SEARCH)
    print(
        f"TCG grid: orth_lambda in {TCG_ORTH_LAMBDA_SEARCH}, "
        f"num_patterns in {TCG_NUM_PATTERNS_SEARCH} "
        f"({tcg_combos} combos per config)"
    )

    for model_name, dataset_name, num_features, input_len, output_len in MISSING_TCG_EXPERIMENTS:
        for tcg_num_patterns in TCG_NUM_PATTERNS_SEARCH:
            for tcg_orth_lambda in TCG_ORTH_LAMBDA_SEARCH:
                p = Process(
                    target=worker_task_tcg,
                    args=(
                        gpu_queue,
                        model_name,
                        dataset_name,
                        num_features,
                        input_len,
                        output_len,
                        tcg_num_patterns,
                        tcg_orth_lambda,
                    ),
                )
                p.start()
                processes.append(p)
                time.sleep(0.1)

    for model_name, dataset_name, num_features, input_len, output_len in MISSING_RAW_EXPERIMENTS:
        p = Process(
            target=worker_task_raw,
            args=(
                gpu_queue,
                model_name,
                dataset_name,
                num_features,
                input_len,
                output_len,
            ),
        )
        p.start()
        processes.append(p)
        time.sleep(0.1)

    print(f"Scheduled {len(processes)} tasks ({len(MISSING_TCG_EXPERIMENTS) * tcg_combos} TCG + {len(MISSING_RAW_EXPERIMENTS)} RAW)")

    for p in processes:
        p.join()

    print("All experiments finished.")
