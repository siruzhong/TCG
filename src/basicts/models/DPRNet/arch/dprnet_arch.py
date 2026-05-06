from typing import Dict, Optional

import torch
from basicts.modules.embed import PatchEmbedding
from basicts.modules.norm import RevIN
from basicts.modules.dpr import TemporalContextualGating, dpr_orthogonal_loss
from torch import nn

from ..config.dprnet_config import DPRNetConfig


class DPRBlock(nn.Module):
    
    def __init__(self, config: DPRNetConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        
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
        
        self.use_dpr = getattr(config, 'use_dpr', True)
        if self.use_dpr:
            self.dpr = TemporalContextualGating(
                d_model=config.hidden_size,
                num_patterns=config.num_patterns,
                use_multiscale=config.use_multiscale,
                identity_init=config.identity_init,
            )
        
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)
    
    def forward(self, x: torch.Tensor, return_aux: bool = False):
        residual = x
        x = self.norm1(x)
        x = self.base_mapping(x)
        x = x + residual
        
        x = self.norm2(x)
        if self.use_dpr:
            x, aux = self.dpr(x, return_aux=return_aux)
        else:
            aux = None
        
        if return_aux:
            return x, aux
        return x


class DPRNetBackbone(nn.Module):
    
    def __init__(self, config: DPRNetConfig):
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
        
        self.dpr_block = DPRBlock(config)
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embedding(inputs)
        
        hidden_states, aux = self.dpr_block(hidden_states, return_aux=True)
        
        hidden_states = hidden_states.reshape(
            -1, self.num_features, self.seq_len, self.hidden_size
        )
        
        return hidden_states, aux


class DPRNetHead(nn.Module):
    
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


class DPRNetForForecasting(nn.Module):
    
    def __init__(self, config: DPRNetConfig):
        super().__init__()
        self.config = config
        
        self.use_revin = config.use_revin
        if self.use_revin:
            self.revin = RevIN(
                num_features=config.num_features,
                affine=config.affine,
                subtract_last=config.subtract_last
            )
        
        self.backbone = DPRNetBackbone(config)
        
        self.flatten = nn.Flatten(start_dim=-2)
        self.head = DPRNetHead(
            input_size=self.backbone.seq_len * config.hidden_size,
            output_size=config.output_len,
            individual=config.individual_head,
            num_features=config.num_features,
            dropout=config.head_dropout
        )
        
        self.orth_lambda = config.orth_lambda
    
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
        
        if aux and "routing_probs" in aux:
            output["routing_probs"] = aux["routing_probs"]
        
        if self.orth_lambda > 0 and self.backbone.dpr_block.use_dpr:
            output["dpr_orth"] = self.orth_lambda * dpr_orthogonal_loss(
                self.backbone.dpr_block.dpr.mode_table
            )
        
        return output
