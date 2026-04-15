#!/usr/bin/env python3
"""
RQ4 Visualization: Extract and visualize routing probabilities from TCG modules.

This script demonstrates that TCG learns to identify different physical dynamics
by visualizing the routing probabilities across different time series patterns.

Usage:
    python run_rq4_visualization.py --checkpoint <path> --dataset <name> --output <dir>
    python run_rq4_visualization.py --checkpoint checkpoints/xxx/model.pt --dataset Illness --output ./rq4_vis
"""
import argparse
import os
import sys
from typing import Dict, List

import numpy as np
import torch

script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from basicts.vis.rq4_pattern_grid import (
    aggregate_routing_to_timestep,
    extract_top_segments,
    compute_pattern_stats,
    visualize_pattern_segment_grid,
    visualize_activation_timeline,
    visualize_transition_matrix,
)


class RoutingCaptor:
    """Context manager to capture routing probabilities from TCG modules."""
    
    def __init__(self, model):
        self.model = model
        self.routing_probs: List[torch.Tensor] = []
        self.hooks: List = []
        self._orig_forwards: Dict = {}
        
    def _patch_tcg_forward(self, tcg_module):
        """Patch TCG forward to always return aux with routing_probs."""
        orig_forward = tcg_module.forward
        
        def patched_forward(x, return_aux=False):
            result, aux = orig_forward(x, return_aux=True)
            if isinstance(aux, dict) and "routing_probs" in aux:
                self.routing_probs.append(aux["routing_probs"].detach().cpu())
            return result
        
        self._orig_forwards[id(tcg_module)] = orig_forward
        tcg_module.forward = patched_forward
        
    def _restore_tcg_forward(self, tcg_module):
        """Restore original TCG forward."""
        orig_id = id(tcg_module)
        if orig_id in self._orig_forwards:
            tcg_module.forward = self._orig_forwards[orig_id]
            del self._orig_forwards[orig_id]
    
    def __enter__(self):
        from basicts.modules.tcg import TemporalContextualGating
        for name, module in self.model.named_modules():
            if isinstance(module, TemporalContextualGating):
                self._patch_tcg_forward(module)
        return self
    
    def __exit__(self, *args):
        from basicts.modules.tcg import TemporalContextualGating
        for name, module in self.model.named_modules():
            if isinstance(module, TemporalContextualGating):
                self._restore_tcg_forward(module)
        self._orig_forwards.clear()


DATASET_CONFIG = {
    "Illness": {"input_len": 24, "output_len": 48, "num_features": 7},
    "ExchangeRate": {"input_len": 336, "output_len": 96, "num_features": 8},
}


def main():
    parser = argparse.ArgumentParser(description="RQ4 Visualization for TCG Routing Probabilities")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--dataset", type=str, default="Illness", choices=["Illness", "ExchangeRate"],
                        help="Dataset name")
    parser.add_argument("--output", type=str, default="./rq4_visualizations", help="Output directory")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"],
                        help="Data split to use")
    parser.add_argument("--model", type=str, default="PatchTST", choices=["PatchTST", "TimeFilter", "WPMixer"],
                        help="Model type")
    parser.add_argument("--k", type=int, default=8, help="TCG num_patterns (only for TCGConfig)")
    parser.add_argument("--samples", type=int, default=5, help="Number of samples to visualize")
    args = parser.parse_args()
    
    dataset_cfg = DATASET_CONFIG[args.dataset]
    input_len = dataset_cfg["input_len"]
    output_len = dataset_cfg["output_len"]
    num_features = dataset_cfg["num_features"]
    
    os.makedirs(args.output, exist_ok=True)
    
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    
    if "model" in ckpt:
        state_dict = ckpt["model"]
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if args.model == "PatchTST":
        from basicts.models.PatchTST import PatchTSTForForecasting, PatchTSTConfig
        from basicts.configs import TCGConfig
        
        cfg = PatchTSTConfig(
            input_len=input_len,
            output_len=output_len,
            num_features=num_features,
        )
        cfg.tcg = TCGConfig(
            enabled=True,
            num_patterns=args.k,
            orth_lambda=0.0001,
        )
        model = PatchTSTForForecasting(cfg)
    elif args.model == "TimeFilter":
        from basicts.models.TimeFilter import TimeFilterForForecasting, TimeFilterConfig
        from basicts.configs import TCGConfig
        
        cfg = TimeFilterConfig(
            input_len=input_len,
            output_len=output_len,
            num_features=num_features,
        )
        cfg.tcg = TCGConfig(
            enabled=True,
            num_patterns=args.k,
            orth_lambda=0.0001,
        )
        model = TimeFilterForForecasting(cfg)
    elif args.model == "WPMixer":
        from basicts.models.WPMixer import WPMixerForForecasting, WPMixerConfig
        from basicts.configs import TCGConfig
        
        cfg = WPMixerConfig(
            input_len=input_len,
            output_len=output_len,
            num_features=num_features,
        )
        cfg.tcg = TCGConfig(
            enabled=True,
            num_patterns=args.k,
            orth_lambda=0.0001,
        )
        model = WPMixerForForecasting(cfg)
    
    try:
        model.load_state_dict(state_dict, strict=False)
        print("Checkpoint loaded successfully")
    except Exception as e:
        print(f"Warning: Could not load state dict strictly: {e}")
    
    model.to(device)
    model.eval()
    
    print(f"Loading dataset: {args.dataset}")
    from basicts.data import BasicTSForecastingDataset
    from torch.utils.data import DataLoader
    from basicts.utils.constants import BasicTSMode
    
    dataset = BasicTSForecastingDataset(
        dataset_name=args.dataset,
        input_len=input_len,
        output_len=output_len,
        mode=BasicTSMode.TEST,
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.samples,
        num_workers=0,
        shuffle=False,
    )
    
    print("Running inference to capture routing probabilities...")
    
    all_routing = []
    all_time_series = []
    
    with RoutingCaptor(model) as captor:
        for batch_idx, batch in enumerate(dataloader):
            inputs = batch["inputs"].to(device)
            
            with torch.no_grad():
                _ = model(inputs)
            
            if captor.routing_probs:
                all_routing.append(torch.cat(captor.routing_probs, dim=0))
                all_time_series.append(inputs.cpu().numpy())
    
    if not all_routing:
        print("ERROR: No routing probabilities captured. Is TCG enabled in the model?")
        return
    
    routing_probs = torch.cat(all_routing, dim=0).numpy()
    time_series_batch = np.concatenate(all_time_series, axis=0)
    
    print(f"Captured routing_probs shape: {routing_probs.shape}")
    print(f"Time series shape: {time_series_batch.shape}")
    
    # routing_probs shape: (B*N, P, K) where B=batch, N=num_features, P=num_patches, K=num_patterns
    # time_series_batch shape: (B, input_len, num_features)
    
    B, L, N = time_series_batch.shape
    K = args.k
    patch_len = 16
    stride = 8
    
    # Reshape routing_probs: (B*N, P, K) -> (B, N, P, K)
    N_actual = routing_probs.shape[0] // B
    P = routing_probs.shape[1]
    routing_reshaped = routing_probs.reshape(B, N_actual, P, K)
    
    # Aggregate patch-level to timestep-level using new API
    routing_timestep = aggregate_routing_to_timestep(
        routing_reshaped, input_len, patch_len=patch_len, stride=stride, method='weighted'
    )
    
    segments = extract_top_segments(routing_timestep, time_series_batch, N=3)
    stats = compute_pattern_stats(segments)
    
    dataset_name = args.dataset
    
    if dataset_name == "Illness":
        prefix = "ili"
    else:
        prefix = "exchange"
    
    visualize_pattern_segment_grid(
        segments, stats,
        os.path.join(args.output, f"{prefix}_pattern_segments"),
        dataset_name
    )
    
    visualize_activation_timeline(
        time_series_batch[0], routing_timestep[0],
        os.path.join(args.output, f"{prefix}_activation_timeline"),
        dataset_name
    )
    
    visualize_transition_matrix(
        routing_timestep[0],
        os.path.join(args.output, f"{prefix}_transition_matrix"),
        dataset_name
    )
    
    print(f"\n=== RQ4 Visualization Complete ===")
    print(f"Dataset: {args.dataset}")
    print(f"Outputs saved to: {args.output}")
    print(f"Generated:")
    print(f"  - {prefix}_pattern_segments.png/pdf")
    print(f"  - {prefix}_activation_timeline.png/pdf")
    print(f"  - {prefix}_transition_matrix.png/pdf")


if __name__ == "__main__":
    main()