from dataclasses import dataclass, field

from peft.config import PromptLearningConfig
from tuners.base.config import GraphEncoderConfig

@dataclass
class GraphPromptEncoderConfig(PromptLearningConfig, GraphEncoderConfig):
    """
    This class stores the configuration for cross_modal_alignment_module in [`GraphPromptEncoder`].
    """
    cma_hidden_size: int = field(
        default=None,
        metadata={"help": "The hidden size of the multi-head self-attention."}, 
    )
    cma_num_heads: int = field(
        default=32,
        metadata={"help": "The number of attention heads for query projection in the multi-head self-attention."},
    )
    cma_num_key_value_heads: int = field(
        default=32,
        metadata={"help": "The number of attention heads for the key and value projections in the multi-head self-attention."},
    )
    cma_dropout: float = field(
        default=0.1,
        metadata={"help": "The dropout probability for the attention weights in the multi-head self-attention."},
    )
    cma_bias: bool = field(
        default=False,
        metadata={"help": "Whether to use bias in the projection layer of the multi-head self-attention."},
    )
    ablate_cross_modal_alignment_module: bool = field(
        default=False,
        metadata={"help": "Whether to ablate the cross-modal alignment module."},
    )
    ablate_multi_head_attn: bool = field(
        default=False,
        metadata={"help": "Whether to ablate the multi-head attention in the cross-modal alignment module."},
    )
    ablate_projector: bool = field(
        default=False,
        metadata={"help": "Whether to ablate the projector in the cross-modal alignment module."},
    )