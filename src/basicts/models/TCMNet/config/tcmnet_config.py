from dataclasses import dataclass, field

from basicts.configs import BasicTSModelConfig, TCGConfig


@dataclass
class TCMNetConfig(BasicTSModelConfig):
    """
    Config class for TCM-Net (Temporal Context Modulation Network).
    
    TCM-Net: A minimal MLP-based backbone with adaptive Temporal Context Modulation (TCM).
    Replaces heavy Transformer encoders with a lightweight MLP + TCM block.
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
    
    # TCM parameters
    num_patterns: int = field(default=8, metadata={"help": "Number of patterns in TCM."})
    use_multiscale: bool = field(default=True, metadata={"help": "Whether to use multi-scale context in TCM."})
    identity_init: bool = field(default=True, metadata={"help": "Whether to use identity initialization for TCM."})
    orth_lambda: float = field(default=0.01, metadata={"help": "Orthogonality regularization weight."})
    
    # Head parameters
    head_dropout: float = field(default=0.0, metadata={"help": "Dropout rate for head layers."})
    individual_head: bool = field(default=False, metadata={"help": "Whether to use individual head per channel."})
    
    # RevIN parameters
    use_revin: bool = field(default=True, metadata={"help": "Whether to use RevIN."})
    affine: bool = field(default=True, metadata={"help": "Whether to use affine transformation in RevIN."})
    subtract_last: bool = field(default=False, metadata={"help": "Whether to subtract the last element in RevIN."})
    
    # Output
    output_attentions: bool = field(default=False, metadata={"help": "Whether to output attention weights (not used in TCM-Net)."})
