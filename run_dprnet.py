import sys
import os
import time
from multiprocessing import Process, Queue

script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from basicts.models.DPRNet import DPRNetForForecasting, DPRNetConfig
from basicts.configs import BasicTSForecastingConfig
from basicts.runners.callback import EarlyStopping
from basicts import BasicTSLauncher

AVAILABLE_GPUS = [0, 1, 2, 3, 4, 5, 6, 7]
JOBS_PER_GPU = 4

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
    "Illness":  {"input_lens": [24], "output_lens": [24, 36, 48, 60]},
    "COVID19":  {"input_lens": [36], "output_lens": [7, 14, 28, 60]},
    "NABCPU":   {"input_lens": [96], "output_lens": [24, 48, 96, 192]},
    "Sunspots": {"input_lens": [36], "output_lens": [12, 24, 48, 96]},
    "default":  {"input_lens": [96], "output_lens": [96, 192, 336, 720]},
}

# DPR-Net hyperparameter search grid (expanded)
NUM_PATTERNS_SEARCH = [4, 8]
ORTH_LAMBDA_SEARCH = [1e-4]
HIDDEN_SIZE_SEARCH = [64, 128, 256]
PATCH_LEN_SEARCH = [16]
MLP_EXPANSION_SEARCH = [1.0, 2.0, 4.0]
NUM_MLP_LAYERS_SEARCH = [2, 3]
USE_MULTISCALE_SEARCH = [True, False]
IDENTITY_INIT_SEARCH = [False]

CHECKPOINT_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")


def _check_results_exist(dataset_name, input_len, output_len, num_patterns, orth_lambda,
                         hidden_size, patch_len, mlp_expansion, num_mlp_layers, use_multiscale, identity_init):
    base_dir = os.path.join(CHECKPOINT_BASE, "DPRNet", f"{dataset_name}_100_{input_len}_{output_len}")
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
            model_params = cfg.get("model_config", {})
            cfg_num_patterns = model_params.get("num_patterns", 0)
            cfg_orth_lambda = model_params.get("orth_lambda", 0)
            cfg_hidden_size = model_params.get("hidden_size", 0)
            cfg_patch_len = model_params.get("patch_len", 0)
            cfg_mlp_expansion = model_params.get("mlp_expansion", 0)
            cfg_num_mlp_layers = model_params.get("num_mlp_layers", 0)
            cfg_use_multiscale = model_params.get("use_multiscale", False)
            cfg_identity_init = model_params.get("identity_init", True)

            if (cfg_num_patterns == num_patterns and
                cfg_orth_lambda == orth_lambda and
                cfg_hidden_size == hidden_size and
                cfg_patch_len == patch_len and
                cfg_mlp_expansion == mlp_expansion and
                cfg_num_mlp_layers == num_mlp_layers and
                cfg_use_multiscale == use_multiscale and
                cfg_identity_init == identity_init):
                return True
        except Exception:
            continue
    return False


def _format_value(x: float) -> str:
    if x == 0.0:
        return "0"
    exp = f"{x:.0e}"
    return exp if "e" in exp.lower() else f"{x:g}"


def run_experiment(dataset_name, num_features, input_len, output_len, gpu_id,
                   num_patterns, orth_lambda, hidden_size, patch_len, mlp_expansion,
                   num_mlp_layers, use_multiscale, identity_init):
    cfg = DPRNetConfig(
        input_len=input_len,
        output_len=output_len,
        num_features=num_features,
        patch_len=patch_len,
        patch_stride=patch_len // 2,
        hidden_size=hidden_size,
        num_mlp_layers=num_mlp_layers,
        mlp_expansion=mlp_expansion,
        mlp_dropout=0.1,
        num_patterns=num_patterns,
        use_multiscale=use_multiscale,
        identity_init=identity_init,
        orth_lambda=orth_lambda,
        head_dropout=0.0,
        use_revin=True,
    )

    callbacks = [EarlyStopping(patience=10)]
    if orth_lambda > 0:
        from basicts.runners.callback import AddAuxiliaryLoss
        callbacks.append(AddAuxiliaryLoss(losses=["dpr_orth"]))

    basic_cfg = BasicTSForecastingConfig(
        model=DPRNetForForecasting, model_config=cfg,
        dataset_name=dataset_name, input_len=input_len, output_len=output_len,
        use_timestamps=False, use_clean_targets=False,
        gpus=gpu_id, num_epochs=100, batch_size=64, callbacks=callbacks, seed=42,
        train_data_num_workers=16, val_data_num_workers=16, test_data_num_workers=16,
        train_data_pin_memory=True, val_data_pin_memory=True, test_data_pin_memory=True,
    )
    BasicTSLauncher.launch_training(basic_cfg)


def worker_task(
    gpu_queue,
    dataset_name,
    num_features,
    input_len,
    output_len,
    num_patterns,
    orth_lambda,
    hidden_size,
    patch_len,
    mlp_expansion,
    num_mlp_layers,
    use_multiscale,
    identity_init,
):
    task_id = (
        f"DPRNet | {dataset_name} | {input_len}->{output_len} | "
        f"K={num_patterns} orth={_format_value(orth_lambda)} hid={hidden_size} "
        f"patch={patch_len} mlp={mlp_expansion} layers={num_mlp_layers} "
        f"ms={use_multiscale} ident={identity_init}"
    )

    if _check_results_exist(dataset_name, input_len, output_len, num_patterns, orth_lambda,
                           hidden_size, patch_len, mlp_expansion, num_mlp_layers,
                           use_multiscale, identity_init):
        print(f"[Skip] {task_id} - results already exist")
        return

    gpu_id = None
    try:
        gpu_id = gpu_queue.get()
        print(f"[Start] {task_id} on GPU {gpu_id}")

        run_experiment(
            dataset_name,
            num_features,
            input_len,
            output_len,
            str(gpu_id),
            num_patterns,
            orth_lambda,
            hidden_size,
            patch_len,
            mlp_expansion,
            num_mlp_layers,
            use_multiscale,
            identity_init,
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
        f"Scheduling DPR-Net tasks on GPUs {AVAILABLE_GPUS} "
        f"(up to {max_concurrent} concurrent, JOBS_PER_GPU={JOBS_PER_GPU})"
    )

    total_combos = (len(NUM_PATTERNS_SEARCH) * len(ORTH_LAMBDA_SEARCH) *
                   len(HIDDEN_SIZE_SEARCH) * len(PATCH_LEN_SEARCH) *
                   len(MLP_EXPANSION_SEARCH) * len(NUM_MLP_LAYERS_SEARCH) *
                   len(USE_MULTISCALE_SEARCH) * len(IDENTITY_INIT_SEARCH))
    print(
        f"Hyperparameter grid: "
        f"num_patterns={NUM_PATTERNS_SEARCH}, "
        f"orth_lambda={ORTH_LAMBDA_SEARCH}, "
        f"hidden_size={HIDDEN_SIZE_SEARCH}, "
        f"patch_len={PATCH_LEN_SEARCH}, "
        f"mlp_expansion={MLP_EXPANSION_SEARCH}, "
        f"num_mlp_layers={NUM_MLP_LAYERS_SEARCH}, "
        f"use_multiscale={USE_MULTISCALE_SEARCH}, "
        f"identity_init={IDENTITY_INIT_SEARCH} "
        f"({total_combos} combos per config)"
    )

    total_tasks = 0

    for dataset_name, num_features in DATASETS:
        config = DATASET_CONFIGS.get(dataset_name, DATASET_CONFIGS["default"])

        for input_len in config["input_lens"]:
            for output_len in config["output_lens"]:
                for num_patterns in NUM_PATTERNS_SEARCH:
                    for orth_lambda in ORTH_LAMBDA_SEARCH:
                        for hidden_size in HIDDEN_SIZE_SEARCH:
                            for patch_len in PATCH_LEN_SEARCH:
                                for mlp_expansion in MLP_EXPANSION_SEARCH:
                                    for num_mlp_layers in NUM_MLP_LAYERS_SEARCH:
                                        for use_multiscale in USE_MULTISCALE_SEARCH:
                                            for identity_init in IDENTITY_INIT_SEARCH:
                                                p = Process(
                                                    target=worker_task,
                                                    args=(
                                                        gpu_queue,
                                                        dataset_name,
                                                        num_features,
                                                        input_len,
                                                        output_len,
                                                        num_patterns,
                                                        orth_lambda,
                                                        hidden_size,
                                                        patch_len,
                                                        mlp_expansion,
                                                        num_mlp_layers,
                                                        use_multiscale,
                                                        identity_init,
                                                    ),
                                                )
                                                p.start()
                                                processes.append(p)
                                                time.sleep(0.1)
                                                total_tasks += 1

    print(f"Scheduled {len(processes)} DPR-Net tasks")

    for p in processes:
        p.join()

    print("All DPR-Net experiments finished.")