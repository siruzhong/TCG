"""
Non-TCG baselines: one run per (model, dataset, input_len, output_len) with default model
configs. No hyperparameter search. Skip if that exact training (same config hash) already
finished — uses the launcher’s `cfg.md5` subfolder, not any TCG-specific fields in cfg.
"""
import sys
import os
import time
from multiprocessing import Process, Queue

script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from basicts.models.Autoformer import Autoformer, AutoformerConfig
from basicts.models.Crossformer import Crossformer, CrossformerConfig
from basicts.models.DLinear import DLinear, DLinearConfig
from basicts.models.Informer import Informer, InformerConfig
from basicts.models.iTransformer import iTransformerForForecasting, iTransformerConfig
from basicts.models.PatchTST import PatchTSTForForecasting, PatchTSTConfig
from basicts.models.TimeMixer import TimeMixerForForecasting, TimeMixerConfig
from basicts.models.TimesNet import TimesNetForForecasting, TimesNetConfig
from basicts.models.TimeFilter import TimeFilterForForecasting, TimeFilterConfig
from basicts.models.WPMixer import WPMixerForForecasting, WPMixerConfig
from basicts.configs import BasicTSForecastingConfig
from basicts.runners.callback import EarlyStopping
from basicts import BasicTSLauncher

AVAILABLE_GPUS = [0, 1, 4, 5, 6, 7]
JOBS_PER_GPU = 2

MODELS = [
    "iTransformer",
    "DLinear",
    "Autoformer",
    "WPMixer",
]

DATASETS = [
    ("ETTh1", 7),
    ("ETTh2", 7),
    ("ETTm1", 7),
    ("ETTm2", 7),
    ("Weather", 21),
    ("Illness", 7),
    ("ExchangeRate", 8),
    ("BeijingAirQuality", 7),
    ("COVID19", 8),
    ("VIX", 1),
    ("NABCPU", 3),
    ("Sunspots", 1),
]

DATASET_CONFIGS = {
    "Illness": {"input_lens": [24], "output_lens": [24, 36, 48, 60]},
    "COVID19": {"input_lens": [36], "output_lens": [7, 14, 28, 60]},
    "NABCPU": {"input_lens": [96], "output_lens": [24, 48, 96, 192]},
    "Sunspots": {"input_lens": [36], "output_lens": [12, 24, 48, 96]},
    "default": {"input_lens": [96], "output_lens": [96, 192, 336, 720]},
}

USE_CLEAN_TARGETS = False
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def _raw_results_exist(
    model_name, dataset_name, num_features, input_len, output_len
):
    """True if this script’s training config already has test_metrics (same run id as `cfg.md5`)."""
    cfg = _make_forecasting_config(
        model_name,
        dataset_name,
        num_features,
        input_len,
        output_len,
        gpus="0",
    )
    run_dir = os.path.join(_PROJECT_ROOT, cfg.ckpt_save_dir, cfg.md5)
    return os.path.isfile(os.path.join(run_dir, "test_metrics.json"))


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
    if dataset_name == "BeijingAirQuality":
        return [24, 7, 31, 366]
    if dataset_name in ("COVID19", "VIX"):
        return [1, 7, 31, 366]
    if dataset_name == "NABCPU":
        return [288, 7, 31, 366]
    if dataset_name == "Sunspots":
        return [1, 1, 1, 12]
    return [60, 7, 31, 366]


def get_model_config(model_name, input_len, output_len, num_features, dataset_name):
    if model_name == "Crossformer":
        cfg = CrossformerConfig(
            input_len=input_len, output_len=output_len, num_features=num_features
        )
        return Crossformer, cfg, False
    elif model_name == "Informer":
        cfg = InformerConfig(
            input_len=input_len,
            output_len=output_len,
            label_len=output_len // 2,
            num_features=num_features,
            use_timestamps=True,
            timestamp_sizes=get_timestamp_sizes(dataset_name),
        )
        return Informer, cfg, True
    elif model_name == "DLinear":
        cfg = DLinearConfig(
            input_len=input_len,
            output_len=output_len,
            num_features=num_features,
            individual=False,
        )
        return DLinear, cfg, False
    elif model_name == "Autoformer":
        cfg = AutoformerConfig(
            input_len=input_len,
            output_len=output_len,
            label_len=output_len // 2,
            num_features=num_features,
            use_timestamps=True,
            timestamp_sizes=get_timestamp_sizes(dataset_name),
        )
        return Autoformer, cfg, True
    elif model_name == "iTransformer":
        cfg = iTransformerConfig(
            input_len=input_len, output_len=output_len, num_features=num_features
        )
        return iTransformerForForecasting, cfg, False
    elif model_name == "PatchTST":
        cfg = PatchTSTConfig(
            input_len=input_len, output_len=output_len, num_features=num_features
        )
        return PatchTSTForForecasting, cfg, False
    elif model_name == "TimeMixer":
        cfg = TimeMixerConfig(
            input_len=input_len, output_len=output_len, num_features=num_features
        )
        return TimeMixerForForecasting, cfg, False
    elif model_name == "TimesNet":
        cfg = TimesNetConfig(
            input_len=input_len,
            output_len=output_len,
            num_features=num_features,
            use_timestamps=True,
            timestamp_sizes=get_timestamp_sizes(dataset_name),
        )
        return TimesNetForForecasting, cfg, True
    elif model_name == "TimeFilter":
        cfg = TimeFilterConfig(
            input_len=input_len,
            output_len=output_len,
            num_features=num_features,
            patch_len=6,
        )
        return TimeFilterForForecasting, cfg, False
    elif model_name == "WPMixer":
        cfg = WPMixerConfig(
            input_len=input_len, output_len=output_len, num_features=num_features
        )
        return WPMixerForForecasting, cfg, False
    else:
        raise ValueError(f"Unknown model: {model_name}")


def _make_forecasting_config(
    model_name,
    dataset_name,
    num_features,
    input_len,
    output_len,
    gpus,
):
    model_class, model_config, use_timestamps = get_model_config(
        model_name, input_len, output_len, num_features, dataset_name
    )
    callbacks = [EarlyStopping(patience=10)]
    return BasicTSForecastingConfig(
        model=model_class,
        model_config=model_config,
        dataset_name=dataset_name,
        input_len=input_len,
        output_len=output_len,
        use_timestamps=use_timestamps,
        use_clean_targets=USE_CLEAN_TARGETS,
        gpus=gpus,
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


def run_experiment(model_name, dataset_name, num_features, input_len, output_len, gpu_id):
    cfg = _make_forecasting_config(
        model_name,
        dataset_name,
        num_features,
        input_len,
        output_len,
        gpus=str(gpu_id),
    )
    BasicTSLauncher.launch_training(cfg)


def worker_task_raw(
    gpu_queue,
    model_name,
    dataset_name,
    num_features,
    input_len,
    output_len,
):
    task_id = f"{model_name} | {dataset_name} | {input_len}->{output_len} | RAW"

    if _raw_results_exist(
        model_name, dataset_name, num_features, input_len, output_len
    ):
        print(f"[Skip] {task_id} - results already exist")
        return

    gpu_id = None
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
        f"RAW baselines only — scheduling on GPUs {AVAILABLE_GPUS} "
        f"(up to {max_concurrent} concurrent, JOBS_PER_GPU={JOBS_PER_GPU})"
    )

    total_raw = 0
    for model_name in MODELS:
        for dataset_name, num_features in DATASETS:
            config = DATASET_CONFIGS.get(dataset_name, DATASET_CONFIGS["default"])
            for input_len in config["input_lens"]:
                for output_len in config["output_lens"]:
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

    print(f"Scheduled {len(processes)} tasks ({total_raw} RAW)")

    for p in processes:
        p.join()

    print("All experiments finished.")
