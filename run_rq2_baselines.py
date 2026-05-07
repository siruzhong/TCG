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
from basicts.configs import BasicTSForecastingConfig, DPRConfig
from basicts.runners.callback import AddAuxiliaryLoss, EarlyStopping
from basicts import BasicTSLauncher

AVAILABLE_GPUS = [0, 1, 2, 3, 4, 5, 6, 7]
JOBS_PER_GPU = 8

MODELS = [
    "Informer",
    "Crossformer",
    "PatchTST",
    "TimesNet",
    "TimeMixer",
    "TimeFilter",
    "WPMixer",
    "DPRNet",
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

# Short datasets (few hundred to few thousand points) need short horizons to
# keep enough train samples after windowing. Values were chosen so that
# train_len - input_len - max(output_len) + 1 >= ~200 samples.
DATASET_CONFIGS = {
    "Illness":  {"input_lens": [24], "output_lens": [24, 36, 48, 60]},
    "COVID19":  {"input_lens": [36], "output_lens": [7, 14, 28, 60]},
    "NABCPU":   {"input_lens": [96], "output_lens": [24, 48, 96, 192]},
    "Sunspots": {"input_lens": [36], "output_lens": [12, 24, 48, 96]},
    "default":  {"input_lens": [96], "output_lens": [96, 192, 336, 720]},
}

DPR_ORTH_LAMBDA_SEARCH = [1e-4, 1e-3, 0.0]
DPR_NUM_PATTERNS_SEARCH = [4, 8, 16]
DPR_CONV_KERNELS_SEARCH = [(1,), (3,), (7,)]
USE_CLEAN_TARGETS = False

# Patch-based models benefit from disabling DPR's multi-scale depthwise conv
# (use k=1 point-wise only). Patching already encodes local temporal structure,
# so the k=3/7 kernels tend to interfere rather than help.
PATCH_MODELS_NO_CONV = {"PatchTST", "WPMixer", "TimeFilter"}

CHECKPOINT_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")


def _check_results_exist(
    model_name,
    dataset_name,
    input_len,
    output_len,
    enable_dpr,
    dpr_num_patterns=None,
    dpr_orth_lambda=None,
    dpr_conv_kernels=None,
):
    base_dir = os.path.join(CHECKPOINT_BASE, model_name, f"{dataset_name}_100_{input_len}_{output_len}")
    if not os.path.exists(base_dir):
        return False
    
    for subdir in os.listdir(base_dir):
        full_path = os.path.join(base_dir, subdir)
        if not os.path.isdir(full_path):
            continue
        metrics_file = os.path.join(full_path, "test_metrics.json")
        cfg_file = os.path.join(full_path, "cfg.json")
        if not os.path.exists(metrics_file) or not os.path.exists(cfg_file):
            continue
        try:
            import json
            with open(cfg_file, 'r') as f:
                cfg = json.load(f)
            dpr_params = cfg.get("model_config", {}).get("dpr", {}).get("params", {})
            cfg_enabled = dpr_params.get("enabled", "")
            is_dpr_enabled = str(cfg_enabled).lower() == "true"
            if enable_dpr != is_dpr_enabled:
                continue
            if enable_dpr and dpr_num_patterns is not None:
                cfg_num_patterns = int(dpr_params.get("num_patterns", 0))
                cfg_orth_lambda = float(dpr_params.get("orth_lambda", 0))
                cfg_conv_kernels = dpr_params.get("conv_kernels", None)
                # Stored as list in JSON; normalize to tuple[int, ...] or None.
                if cfg_conv_kernels is not None:
                    try:
                        cfg_conv_kernels = tuple(int(k) for k in cfg_conv_kernels)
                    except Exception:
                        cfg_conv_kernels = None

                if (
                    cfg_num_patterns != dpr_num_patterns
                    or cfg_orth_lambda != dpr_orth_lambda
                    or cfg_conv_kernels != dpr_conv_kernels
                ):
                    continue
            return True
        except Exception:
            continue
    return False


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
            num_features=num_features,
            patch_len=6,
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
    if kwargs.get("enable_dpr", True) and hasattr(model_config, "dpr"):
        use_multiscale = model_name not in PATCH_MODELS_NO_CONV
        dpr_cfg = DPRConfig(
            enabled=True,
            num_patterns=int(kwargs.get("dpr_num_patterns", kwargs.get("dpr_K", 8))),
            orth_lambda=float(kwargs.get("dpr_orth_lambda", 0.01)),
            use_multiscale=use_multiscale,
            conv_kernels=tuple(kwargs["dpr_conv_kernels"])
            if "dpr_conv_kernels" in kwargs and kwargs["dpr_conv_kernels"] is not None
            else None,
        )
        model_config.dpr = dpr_cfg
        if dpr_cfg.orth_lambda > 0:
            callbacks.append(AddAuxiliaryLoss(losses=["dpr_orth"]))

    cfg = BasicTSForecastingConfig(
        model=model_class, model_config=model_config,
        dataset_name=dataset_name, input_len=input_len, output_len=output_len,
        use_timestamps=use_timestamps, use_clean_targets=USE_CLEAN_TARGETS,
        gpus=gpu_id, num_epochs=100, batch_size=64, callbacks=callbacks, seed=42,
        train_data_num_workers=16, val_data_num_workers=16, test_data_num_workers=16,
        train_data_pin_memory=True, val_data_pin_memory=True, test_data_pin_memory=True,
    )
    BasicTSLauncher.launch_training(cfg)


def worker_task_dpr(
    gpu_queue,
    model_name,
    dataset_name,
    num_features,
    input_len,
    output_len,
    dpr_num_patterns,
    dpr_orth_lambda,
    dpr_conv_kernels,
):
    effective_kernels = (1,) if model_name in PATCH_MODELS_NO_CONV else tuple(int(k) for k in dpr_conv_kernels)
    kernels_tag = "k" + "-".join(str(k) for k in effective_kernels)
    task_id = (
        f"{model_name} | {dataset_name} | {input_len}->{output_len} | "
        f"DPR[{kernels_tag}] K={dpr_num_patterns} orth={_format_orth_lambda(dpr_orth_lambda)}"
    )

    if _check_results_exist(
        model_name,
        dataset_name,
        input_len,
        output_len,
        True,
        dpr_num_patterns,
        dpr_orth_lambda,
        effective_kernels,
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
            enable_dpr=True,
            dpr_num_patterns=dpr_num_patterns,
            dpr_orth_lambda=dpr_orth_lambda,
            dpr_conv_kernels=effective_kernels,
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
    task_id = f"{model_name} | {dataset_name} | {input_len}->{output_len} | RAW"

    if _check_results_exist(model_name, dataset_name, input_len, output_len, False):
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
            enable_dpr=False,
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

    dpr_combos = len(DPR_NUM_PATTERNS_SEARCH) * len(DPR_ORTH_LAMBDA_SEARCH)
    kernels_combos = len(DPR_CONV_KERNELS_SEARCH)
    print(
        f"DPR grid: orth_lambda in {DPR_ORTH_LAMBDA_SEARCH}, "
        f"num_patterns in {DPR_NUM_PATTERNS_SEARCH} "
        f"conv_kernels in {DPR_CONV_KERNELS_SEARCH} "
        f"({dpr_combos * kernels_combos} combos per config)"
    )

    total_dpr = 0
    total_raw = 0

    for model_name in MODELS:
        for dataset_name, num_features in DATASETS:
            config = DATASET_CONFIGS.get(dataset_name, DATASET_CONFIGS["default"])

            for input_len in config["input_lens"]:
                for output_len in config["output_lens"]:
                    for dpr_num_patterns in DPR_NUM_PATTERNS_SEARCH:
                        for dpr_orth_lambda in DPR_ORTH_LAMBDA_SEARCH:
                            kernels_list = [(1,)] if model_name in PATCH_MODELS_NO_CONV else DPR_CONV_KERNELS_SEARCH
                            for dpr_conv_kernels in kernels_list:
                                p = Process(
                                    target=worker_task_dpr,
                                    args=(
                                        gpu_queue,
                                        model_name,
                                        dataset_name,
                                        num_features,
                                        input_len,
                                        output_len,
                                        dpr_num_patterns,
                                        dpr_orth_lambda,
                                        dpr_conv_kernels,
                                    ),
                                )
                                p.start()
                                processes.append(p)
                                time.sleep(0.1)
                                total_dpr += 1

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

    print(f"Scheduled {len(processes)} tasks ({total_dpr} DPR + {total_raw} RAW)")

    for p in processes:
        p.join()

    print("All experiments finished.")
