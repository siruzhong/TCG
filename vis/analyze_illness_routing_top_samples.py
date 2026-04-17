#!/usr/bin/env python3
"""Find Illness test samples with strong, switching TCG routing."""

import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from basicts.configs import TCGConfig
from basicts.data import BasicTSForecastingDataset
from basicts.models.PatchTST import PatchTSTConfig, PatchTSTForForecasting
from basicts.modules.tcg import TemporalContextualGating
from basicts.utils.constants import BasicTSMode


CHECKPOINT_PATH = os.path.join(
    PROJECT_ROOT,
    "checkpoints/test_logits/9378b912cebedfed27f6fff60f402473/PatchTSTForForecasting_best_val_MAE.pt",
)
DATASET_NAME = "Illness"
INPUT_LEN = 24
OUTPUT_LEN = 24
NUM_FEATURES = 7
NUM_PATTERNS = 8
PATCH_LEN = 16
PATCH_STRIDE = 8
BATCH_SIZE = 64
BASELINE_UNIFORM = 1.0 / NUM_PATTERNS


class RoutingCaptor:
    def __init__(self, model):
        self.model = model
        self.routing_probs = []
        self._orig_forwards = {}

    def _patch(self, module):
        orig_forward = module.forward

        def patched(x, return_aux=False):
            result, aux = orig_forward(x, return_aux=True)
            if isinstance(aux, dict) and "routing_probs" in aux:
                self.routing_probs.append(aux["routing_probs"].detach().cpu())
            if return_aux:
                return result, aux
            return result

        self._orig_forwards[id(module)] = orig_forward
        module.forward = patched

    def __enter__(self):
        for _, module in self.model.named_modules():
            if isinstance(module, TemporalContextualGating):
                self._patch(module)
        return self

    def __exit__(self, *args):
        for _, module in self.model.named_modules():
            if isinstance(module, TemporalContextualGating) and id(module) in self._orig_forwards:
                module.forward = self._orig_forwards[id(module)]


def aggregate_patch_routing_to_timestep(routing, input_len, patch_len, patch_stride):
    """Map patch-level routing [B, N, P, K] to timestep-level [B, N, L, K]."""
    batch, num_features, num_patches, num_patterns = routing.shape
    timestep_routing = np.zeros((batch, num_features, input_len, num_patterns), dtype=np.float32)
    weights = np.zeros((batch, num_features, input_len, 1), dtype=np.float32)

    for patch_idx in range(num_patches):
        start = patch_idx * patch_stride
        end = min(start + patch_len, input_len)
        if end <= start:
            continue
        timestep_routing[:, :, start:end, :] += routing[:, :, patch_idx:patch_idx + 1, :]
        weights[:, :, start:end, :] += 1.0

    return timestep_routing / np.maximum(weights, 1e-6)


def build_model(device):
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    state_dict = checkpoint["model_state_dict"]

    cfg = PatchTSTConfig(
        input_len=INPUT_LEN,
        output_len=OUTPUT_LEN,
        num_features=NUM_FEATURES,
        patch_len=PATCH_LEN,
        patch_stride=PATCH_STRIDE,
    )
    cfg.tcg = TCGConfig(
        enabled=True,
        num_patterns=NUM_PATTERNS,
        orth_lambda=1e-4,
        use_multiscale=True,
        identity_init=True,
        discrete_topk=1,
    )

    model = PatchTSTForForecasting(cfg)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint: {CHECKPOINT_PATH}")
    print(f"Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
    model.to(device)
    model.eval()
    return model


def load_test_loader():
    dataset = BasicTSForecastingDataset(
        dataset_name=DATASET_NAME,
        input_len=INPUT_LEN,
        output_len=OUTPUT_LEN,
        mode=BasicTSMode.TEST,
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"Illness test samples: {len(dataset)}")
    return loader


def run_inference_and_capture(model, loader, device):
    all_inputs = []
    all_routing = []

    with RoutingCaptor(model) as captor:
        for batch in loader:
            inputs = batch["inputs"].to(device)
            with torch.no_grad():
                _ = model(inputs)

            if captor.routing_probs:
                batch_routing = torch.cat(captor.routing_probs, dim=0)
                all_routing.append(batch_routing.numpy())
                captor.routing_probs.clear()
            all_inputs.append(inputs.cpu().numpy())

    inputs_np = np.concatenate(all_inputs, axis=0)
    routing_np = np.concatenate(all_routing, axis=0)
    num_samples = inputs_np.shape[0]
    routing_np = routing_np.reshape(num_samples, NUM_FEATURES, -1, NUM_PATTERNS)
    timestep_routing = aggregate_patch_routing_to_timestep(
        routing_np,
        input_len=INPUT_LEN,
        patch_len=PATCH_LEN,
        patch_stride=PATCH_STRIDE,
    )

    print(f"Captured inputs shape: {inputs_np.shape}")
    print(f"Captured routing shape: {routing_np.shape}")
    print(f"Timestep routing shape: {timestep_routing.shape}")
    return inputs_np, timestep_routing


def score_samples(timestep_routing):
    records = []
    for sample_idx in range(timestep_routing.shape[0]):
        for feature_idx in range(timestep_routing.shape[1]):
            routing = timestep_routing[sample_idx, feature_idx]
            dominant = routing.argmax(axis=-1)
            switches = int(np.sum(np.diff(dominant) != 0))
            max_prob = float(routing.max())
            mean_max_prob = float(routing.max(axis=-1).mean())
            score = (max_prob - BASELINE_UNIFORM) * switches
            records.append(
                {
                    "sample_idx": sample_idx,
                    "feature_idx": feature_idx,
                    "score": score,
                    "switches": switches,
                    "max_prob": max_prob,
                    "mean_max_prob": mean_max_prob,
                    "dominant": dominant,
                    "routing": routing,
                    "high_conf": max_prob > 0.5,
                    "clear_regime": len(np.unique(dominant)) >= 3,
                }
            )
    records.sort(key=lambda x: x["score"], reverse=True)
    return records


def print_top_samples(records, top_k=5):
    print("\nTop 5 samples by score = (max_routing_prob - 0.125) * switches")
    for rank, rec in enumerate(records[:top_k], start=1):
        print(
            f"{rank}. sample={rec['sample_idx']}, feature={rec['feature_idx']}, "
            f"score={rec['score']:.4f}, switches={rec['switches']}, "
            f"max_prob={rec['max_prob']:.4f}, mean_max_prob={rec['mean_max_prob']:.4f}, "
            f"high_conf={rec['high_conf']}, clear_regime={rec['clear_regime']}, "
            f"unique_patterns={len(np.unique(rec['dominant']))}"
        )


def print_top_sample_details(record):
    print("\nTop sample routing details")
    print(
        f"sample={record['sample_idx']}, feature={record['feature_idx']}, "
        f"score={record['score']:.4f}, switches={record['switches']}, max_prob={record['max_prob']:.4f}"
    )
    print(f"dominant_patterns={record['dominant'].tolist()}")
    np.set_printoptions(precision=4, suppress=True, linewidth=160)
    print("routing_probs [L, K] =")
    print(record["routing"])
    print("routing_probs transposed for heatmap [K, L] =")
    print(record["routing"].T)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = build_model(device)
    loader = load_test_loader()
    _, timestep_routing = run_inference_and_capture(model, loader, device)
    records = score_samples(timestep_routing)

    qualified = [
        rec
        for rec in records
        if rec["high_conf"] and rec["switches"] >= 2 and rec["clear_regime"]
    ]
    print(f"\nQualified samples meeting all criteria: {len(qualified)}")

    top_records = qualified[:5] if qualified else records[:5]
    print_top_samples(top_records, top_k=min(5, len(top_records)))
    if top_records:
        print_top_sample_details(top_records[0])


if __name__ == "__main__":
    main()
