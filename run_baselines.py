import sys
import os
import time
from multiprocessing import Process, Queue

# Add src directory
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Import Models
from basicts.models.Crossformer import Crossformer, CrossformerConfig
from basicts.models.Informer import Informer, InformerConfig
from basicts.models.iTransformer import iTransformerForForecasting, iTransformerConfig
from basicts.models.PatchTST import PatchTSTForForecasting, PatchTSTConfig
from basicts.models.TimeMixer import TimeMixerForForecasting, TimeMixerConfig
from basicts.models.TimesNet import TimesNetForForecasting, TimesNetConfig
from basicts.configs import BasicTSForecastingConfig, TCGConfig
from basicts.runners.callback import AddAuxiliaryLoss, EarlyStopping
from basicts import BasicTSLauncher

# --- Global Configurations ---
AVAILABLE_GPUS = [0, 1, 2, 3, 4, 5, 6, 7]

# iTransformer omitted from this sweep: iTransformerConfig has no ``tcg`` field and the
# backbone layout is [B, num_features, d] (variate tokens), not a time-major [B, L, d] TCG path.
MODELS = [
    "Informer",
    "Crossformer",
    "PatchTST",
    "TimesNet",
    "TimeMixer",
]
DATASETS = [
    ("ETTh1", 7),
    ("ETTh2", 7),
    ("ETTm1", 7),
    ("ETTm2", 7),
    ("Electricity", 321),
    ("Weather", 21),
    ("Illness", 7),
]

# Length settings
DATASET_CONFIGS = {
    "Illness": {"input_lens": [24], "output_lens": [24, 36, 48, 60]},
    "default": {"input_lens": [96], "output_lens": [96, 192, 336, 720]}
}

USE_CLEAN_TARGETS = True


def get_timestamp_sizes(dataset_name: str):
    """Helper to get timestamp features."""
    if dataset_name in ["ETTh1", "ETTh2"]: return [24, 7, 31, 366]
    if dataset_name in ["ETTm1", "ETTm2", "SyntheticTS"]: return [96, 7, 31, 366]
    if dataset_name.startswith("SyntheticTS"): return [96, 7, 31, 366]
    return [60, 7, 31, 366]


def get_model_config(model_name, input_len, output_len, num_features, dataset_name):
    """Factory to create model class and config using explicit keywords."""

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
    """Setup and launch a single experiment."""
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


def worker_task(gpu_queue, model_name, dataset_name, num_features, input_len, output_len):
    """Worker process: Acquires GPU -> Runs Exp -> Releases GPU."""
    gpu_id = None
    task_id = f"{dataset_name} ({input_len}->{output_len})"

    try:
        gpu_id = gpu_queue.get()
        print(f"[Start] {task_id} on GPU {gpu_id}")

        run_experiment(
            model_name, dataset_name, num_features, input_len, output_len, str(gpu_id),
            enable_tcg=True,
        )

    except Exception as e:
        print(f"[Error] {task_id} failed: {e}")
        import traceback; traceback.print_exc()
    finally:
        if gpu_id is not None:
            gpu_queue.put(gpu_id)
            print(f"[Done] {task_id} released GPU {gpu_id}")


if __name__ == "__main__":
    # Initialize GPU Queue
    gpu_queue = Queue()
    for gpu_id in AVAILABLE_GPUS:
        gpu_queue.put(gpu_id)

    processes = []
    print(f"Scheduling tasks on GPUs: {AVAILABLE_GPUS}")

    # Schedule processes (One process per input/output length combination)
    for model_name in MODELS:
        for dataset_name, num_features in DATASETS:
            config = DATASET_CONFIGS.get(dataset_name, DATASET_CONFIGS["default"])

            for input_len in config["input_lens"]:
                for output_len in config["output_lens"]:
                    p = Process(
                        target=worker_task,
                        args=(gpu_queue, model_name, dataset_name, num_features, input_len, output_len)
                    )
                    p.start()
                    processes.append(p)
                    time.sleep(0.1)

    print(f"Scheduled {len(processes)} tasks.")

    for p in processes:
        p.join()

    print("All experiments finished.")
