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
from basicts.models.TimeFilter import TimeFilterForForecasting, TimeFilterConfig
from basicts.models.WPMixer import WPMixerForForecasting, WPMixerConfig
from basicts.configs import BasicTSForecastingConfig, TCGConfig
from basicts.runners.callback import AddAuxiliaryLoss, EarlyStopping
from basicts import BasicTSLauncher

AVAILABLE_GPUS = [0, 1, 2, 3, 4, 5, 6, 7]
JOBS_PER_GPU = 6

MODELS = [
    "Informer",
    "Crossformer",
    "PatchTST",
    "TimesNet",
    "TimeMixer",
    "TimeFilter",
    "WPMixer",
]

DATASETS = [
    ("ETTh1", 7),
    ("ETTh2", 7),
    ("ETTm1", 7),
    ("ETTm2", 7),
    ("Electricity", 321),
    ("Weather", 21),
    ("Illness", 7),
    ("ExchangeRate", 8),
]

DATASET_CONFIGS = {
    "Illness": {"input_lens": [24], "output_lens": [24, 36, 48, 60]},
    "default": {"input_lens": [96], "output_lens": [96, 192, 336, 720]}
}

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
    elif model_name == "TimeFilter":
        cfg = TimeFilterConfig(
            input_len=input_len,
            output_len=output_len,
            num_features=num_features
        )
        return TimeFilterForForecasting, cfg, False
    elif model_name == "WPMixer":
        cfg = WPMixerConfig(
            input_len=input_len,
            output_len=output_len,
            num_features=num_features
        )
        return WPMixerForForecasting, cfg, False
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

    total_tcg = 0
    total_raw = 0

    for model_name in MODELS:
        for dataset_name, num_features in DATASETS:
            config = DATASET_CONFIGS.get(dataset_name, DATASET_CONFIGS["default"])

            for input_len in config["input_lens"]:
                for output_len in config["output_lens"]:
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
                            total_tcg += 1

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
                    total_raw += 1

    print(f"Scheduled {len(processes)} tasks ({total_tcg} TCG + {total_raw} RAW)")

    for p in processes:
        p.join()

    print("All experiments finished.")
