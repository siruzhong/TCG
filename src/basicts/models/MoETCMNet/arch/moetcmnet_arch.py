from typing import Dict, Optional

import torch
import torch.nn.functional as F
from basicts.modules.embed import PatchEmbedding
from basicts.modules.norm import RevIN
from torch import nn

from ..config.moetcmnet_config import MoETCMNetConfig


class MoELayer(nn.Module):
    """
    Mixture of Experts (MoE) layer implementing sparse top-k gating.
    
    Each token is routed to top-k experts based on a gating network.
    The output is a weighted combination of the selected experts.
    Includes load balancing loss for training stability.
    """
    
    def __init__(
        self,
        d_model: int,
        num_experts: int = 8,
        top_k: int = 1,
        noisy_gating: bool = True,
        expert_hidden_ratio: float = 0.5,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.noisy_gating = noisy_gating
        
        expert_hidden_dim = max(16, int(d_model * expert_hidden_ratio))
        
        # Expert networks: each is a small MLP
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, expert_hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(expert_hidden_dim, d_model),
            )
            for _ in range(num_experts)
        ])
        
        # Gating network
        self.gate = nn.Linear(d_model, num_experts)
        
        # Noise for load balancing during training
        if noisy_gating:
            self.noise_linear = nn.Linear(d_model, num_experts)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.gate.weight, std=0.02)
        nn.init.zeros_(self.gate.bias)
        if self.noisy_gating:
            nn.init.normal_(self.noise_linear.weight, std=0.02)
            nn.init.zeros_(self.noise_linear.bias)
        for expert in self.experts:
            nn.init.normal_(expert[0].weight, std=0.02)
            nn.init.zeros_(expert[0].bias)
            nn.init.normal_(expert[-1].weight, std=0.02)
            nn.init.zeros_(expert[-1].bias)
    
    def _noisy_top_k_gating(
        self,
        x: torch.Tensor,
        training: bool = True,
        noise_epsilon: float = 1e-2,
    ):
        """
        Noisy top-k gating with load balancing.
        
        Args:
            x: [B, L, d_model]
            training: whether in training mode
            noise_epsilon: noise scaling factor
            
        Returns:
            gates: [B, L, num_experts] - gating weights
            load: [num_experts] - fraction of tokens routed to each expert
            importance: [num_experts] - sum of gating weights per expert
        """
        clean_logits = self.gate(x)  # [B, L, num_experts]
        
        if self.noisy_gating and training:
            raw_noise_stddev = self.noise_linear(x)
            noise_stddev = F.softplus(raw_noise_stddev) + noise_epsilon
            noisy_logits = clean_logits + torch.randn_like(clean_logits) * noise_stddev
            logits = noisy_logits
        else:
            logits = clean_logits
        
        # Compute top-k gating
        top_logits, top_indices = logits.topk(min(self.top_k, self.num_experts), dim=-1)
        top_gates = F.softmax(top_logits, dim=-1)
        
        # Create sparse gate matrix
        gates = torch.zeros_like(logits)
        gates.scatter_(-1, top_indices, top_gates)
        
        # Compute load balancing statistics
        load = gates.sum(dim=(0, 1)) / (x.size(0) * x.size(1))  # [num_experts]
        importance = gates.sum(dim=(0, 1)) / (x.size(0) * x.size(1))  # [num_experts]
        
        return gates, load, importance
    
    def forward(
        self,
        x: torch.Tensor,
        return_aux: bool = False,
    ):
        """
        Args:
            x: [B, L, d_model]
            return_aux: If True, return dict with load balancing loss.
            
        Returns:
            out: [B, L, d_model] - MoE output
            aux: Optional dict with 'moe_loss' for load balancing.
        """
        b, l, d = x.shape
        
        # Get gating weights
        gates, load, importance = self._noisy_top_k_gating(x, training=self.training)
        
        # Flatten batch and sequence dimensions
        x_flat = x.reshape(-1, d)  # [B*L, d_model]
        gates_flat = gates.reshape(-1, self.num_experts)  # [B*L, num_experts]
        
        # Compute expert outputs
        out = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            # Get tokens routed to this expert
            expert_mask = gates_flat[:, i] > 0
            if expert_mask.any():
                expert_input = x_flat[expert_mask]
                expert_output = expert(expert_input)
                out[expert_mask] += expert_output * gates_flat[expert_mask, i:i+1]
        
        out = out.reshape(b, l, d)
        
        if not return_aux:
            return out
        
        # Compute load balancing loss
        # Coefficient of variation squared to encourage uniform usage
        cv_importance = ((importance - importance.mean()) ** 2).mean() / (importance.mean() ** 2 + 1e-10)
        cv_load = ((load - load.mean()) ** 2).mean() / (load.mean() ** 2 + 1e-10)
        moe_loss = cv_importance + cv_load
        
        aux = {
            "moe_loss": moe_loss,
            "load": load.detach(),
            "importance": importance.detach(),
            "gates": gates.detach(),
        }
        
        return out, aux


class MoETCMBlock(nn.Module):
    """
    MoE-TCM Block: MLP base mapping + MoE layer.
    
    Architecture is identical to TCMBlock except TCG is replaced with MoE.
    """
    
    def __init__(self, config: MoETCMNetConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        
        # Base mapping MLP (identical to TCMBlock)
        mlp_layers = []
        intermediate_dim = int(config.hidden_size * config.mlp_expansion)
        
        for i in range(config.num_mlp_layers):
            if i == 0:
                mlp_layers.append(nn.Linear(config.hidden_size, intermediate_dim))
            elif i == config.num_mlp_layers - 1:
                mlp_layers.append(nn.Linear(intermediate_dim, config.hidden_size))
            else:
                mlp_layers.append(nn.Linear(intermediate_dim, intermediate_dim))
            
            if i < config.num_mlp_layers - 1:
                if config.mlp_activation == "gelu":
                    mlp_layers.append(nn.GELU())
                elif config.mlp_activation == "relu":
                    mlp_layers.append(nn.ReLU())
                elif config.mlp_activation == "silu":
                    mlp_layers.append(nn.SiLU())
                mlp_layers.append(nn.Dropout(config.mlp_dropout))
        
        self.base_mapping = nn.Sequential(*mlp_layers)
        
        # MoE layer (replaces TCG)
        self.moe = MoELayer(
            d_model=config.hidden_size,
            num_experts=config.num_experts,
            top_k=config.top_k,
            noisy_gating=config.noisy_gating,
            expert_hidden_ratio=config.expert_hidden_ratio,
        )
        
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)
    
    def forward(self, x: torch.Tensor, return_aux: bool = False):
        # Base mapping with residual (identical to TCMBlock)
        residual = x
        x = self.norm1(x)
        x = self.base_mapping(x)
        x = x + residual
        
        # MoE layer with residual (replaces TCG)
        x = self.norm2(x)
        
        if return_aux:
            x, aux = self.moe(x, return_aux=True)
            return x, aux
        
        x = self.moe(x, return_aux=False)
        return x


class MoETCMNetBackbone(nn.Module):
    """
    MoE-TCMNet Backbone: PatchEmbedding + MoE-TCM Block.
    
    Identical to TCMNetBackbone except uses MoETCMBlock.
    """
    
    def __init__(self, config: MoETCMNetConfig):
        super().__init__()
        self.config = config
        self.num_features = config.num_features
        self.hidden_size = config.hidden_size
        
        padding = (0, config.patch_stride) if config.padding else None
        self.embedding = PatchEmbedding(
            hidden_size=config.hidden_size,
            patch_len=config.patch_len,
            stride=config.patch_stride,
            padding=padding,
            dropout=config.mlp_dropout
        )
        self.num_patches = int((config.input_len - config.patch_len) / config.patch_stride + 1)
        if config.padding:
            self.num_patches += 1
        self.seq_len = self.num_patches
        
        self.moe_block = MoETCMBlock(config)
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embedding(inputs)
        
        hidden_states, aux = self.moe_block(hidden_states, return_aux=True)
        
        hidden_states = hidden_states.reshape(
            -1, self.num_features, self.seq_len, self.hidden_size
        )
        
        return hidden_states, aux


class MoETCMNetHead(nn.Module):
    """
    Forecasting head. Identical to TCMNetHead.
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        individual: bool = False,
        num_features: Optional[int] = None,
        dropout: float = 0.0
    ):
        super().__init__()
        self.individual = individual
        self.num_features = num_features
        
        if self.individual:
            if self.num_features is None:
                raise ValueError("num_features is required when individual is True")
            self.linears = nn.ModuleList([
                nn.Linear(input_size, output_size) for _ in range(num_features)
            ])
            self.dropouts = nn.ModuleList([
                nn.Dropout(dropout) for _ in range(num_features)
            ])
        else:
            self.linear = nn.Linear(input_size, output_size)
            self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.individual:
            if self.num_features != x.shape[1]:
                raise ValueError(
                    f"num_features ({self.num_features}) does not match input shape ({x.shape[1]})."
                )
            x_out = []
            for i in range(self.num_features):
                out = self.dropouts[i](self.linears[i](x[:, i, :]))
                x_out.append(out)
            x = torch.stack(x_out, dim=1)
        else:
            x = self.dropout(self.linear(x))
        return x


class MoETCMNetForForecasting(nn.Module):
    """
    MoE-TCMNet for Time Series Forecasting.
    
    Architecture is identical to TCMNetForForecasting except:
    1. Uses MoETCMNetBackbone (MoE instead of TCG)
    2. Returns moe_loss instead of tcg_orth
    """
    
    def __init__(self, config: MoETCMNetConfig):
        super().__init__()
        self.config = config
        
        self.use_revin = config.use_revin
        if self.use_revin:
            self.revin = RevIN(
                num_features=config.num_features,
                affine=config.affine,
                subtract_last=config.subtract_last
            )
        
        self.backbone = MoETCMNetBackbone(config)
        
        self.flatten = nn.Flatten(start_dim=-2)
        self.head = MoETCMNetHead(
            input_size=self.backbone.seq_len * config.hidden_size,
            output_size=config.output_len,
            individual=config.individual_head,
            num_features=config.num_features,
            dropout=config.head_dropout
        )
        
        self.moe_loss_coef = config.moe_loss_coef
    
    def forward(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.use_revin:
            inputs = self.revin(inputs, "norm")
        
        hidden_states, aux = self.backbone(inputs)
        
        hidden_states = self.flatten(hidden_states)
        
        prediction = self.head(hidden_states)
        
        prediction = prediction.transpose(1, 2)
        
        if self.use_revin:
            prediction = self.revin(prediction, "denorm")
        
        output = {"prediction": prediction}
        
        if aux and "gates" in aux:
            output["routing_probs"] = aux["gates"]
        
        if self.moe_loss_coef > 0 and aux and "moe_loss" in aux:
            output["moe_loss"] = self.moe_loss_coef * aux["moe_loss"]
        
        return output
