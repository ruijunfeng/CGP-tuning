from dataclasses import dataclass, field
from typing import List

@dataclass
class GraphEncoderConfig():
    """
    This is the configuration class to store the configuration of a [`GraphEncoder`].
    """
    gnn_input_size: int = field(
        default=None,
        metadata={"help": "The input dimension of node embeddings for the input GAT layer."},
    )
    gnn_hidden_size: int = field(
        default=None,
        metadata={"help": "The hidden dimension of node embeddings for the hidden GAT layer."},
    )
    gnn_output_size: int = field(
        default=None,
        metadata={"help": "The output dimension of node embeddings for the output GAT layer."},
    )
    gnn_num_layers: int = field(
        default=3,
        metadata={"help": "The number of GAT layers."},
    )
    gnn_num_heads: int = field(
        default=4,
        metadata={"help": "The number of attention heads in GAT layers."},
    )    
    gnn_dropout: float = field(
        default=0.0,
        metadata={"help": "The dropout probability for the attention weights in GAT layers."},
    )
    gnn_bias: bool = field(
        default=False,
        metadata={"help": "Whether to include bias in the output of GAT layers."},
    )
    num_node_types: int = field(
        default=18,
        metadata={"help": "The number of node types in the code property graph."},
    )
    num_edge_types: int = field(
        default=13,
        metadata={"help": "The number of edge types in the code property graph."},
    )
    max_num_nodes: int = field(
        default=4096,
        metadata={"help": "The maximum number of nodes to retain in the code property graph of SAGPooling."},
    )
    edge_dim: int = field(
        default=128,
        metadata={"help": "The dimension of edge type embeddings."},
    )
    