from dataclasses import dataclass, field

from basicts.configs import BasicTSModelConfig


@dataclass
class MoEDPRNetConfig(BasicTSModelConfig):
    """
    Config class for MoE-DPRNet (Mixture of Experts Temporal Context Modulation Network).
    
    MoE-DPRNet: A minimal MLP-based backbone with Mixture of Experts (MoE) replacing DPR.
    Replaces the TemporalContextualGating with a sparse MoE layer while keeping all other
    architecture parameters identical to DPRNet.
    """

    input_len: int = field(default=None, metadata={"help": "Input sequence length."})
    output_len: int = field(default=None, metadata={"help": "Output sequence length for forecasting task."})
    num_features: int = field(default=None, metadata={"help": "Number of features."})
    num_classes: int = field(default=None, metadata={"help": "Number of classes for classification task."})
    
    # Patching parameters
    patch_len: int = field(default=16, metadata={"help": "Patch length."})
    patch_stride: int = field(default=8, metadata={"help": "Stride for patching."})
    padding: bool = field(default=True, metadata={"help": "Whether to pad the input sequence before patching."})
    
    # Model dimensions
    hidden_size: int = field(default=256, metadata={"help": "Hidden size / embedding dimension."})
    
    # Base Mapping (MLP) parameters
    num_mlp_layers: int = field(default=2, metadata={"help": "Number of MLP layers in base mapping."})
    mlp_expansion: float = field(default=2.0, metadata={"help": "MLP expansion ratio (intermediate = hidden_size * mlp_expansion)."})
    mlp_dropout: float = field(default=0.1, metadata={"help": "Dropout rate for MLP layers."})
    mlp_activation: str = field(default="gelu", metadata={"help": "Activation function for MLP."})
    
    # MoE parameters (mapped from DPR parameters for fair comparison)
    num_experts: int = field(default=8, metadata={"help": "Number of experts in MoE layer (maps to num_patterns in DPRNet)."})
    top_k: int = field(default=1, metadata={"help": "Top-k experts to route each token to."})
    noisy_gating: bool = field(default=True, metadata={"help": "Whether to add noise to gating for load balancing."})
    moe_loss_coef: float = field(default=0.01, metadata={"help": "Coefficient for MoE load balancing loss (maps to orth_lambda in DPRNet)."})
    expert_hidden_ratio: float = field(default=0.5, metadata={"help": "Expert hidden dim ratio relative to hidden_size."})
    
    # Head parameters
    head_dropout: float = field(default=0.0, metadata={"help": "Dropout rate for head layers."})
    individual_head: bool = field(default=False, metadata={"help": "Whether to use individual head per channel."})
    
    # RevIN parameters
    use_revin: bool = field(default=True, metadata={"help": "Whether to use RevIN."})
    affine: bool = field(default=True, metadata={"help": "Whether to use affine transformation in RevIN."})
    subtract_last: bool = field(default=False, metadata={"help": "Whether to subtract the last element in RevIN."})
    
    # Output
    output_attentions: bool = field(default=False, metadata={"help": "Whether to output attention weights (not used in MoE-DPRNet)."})
