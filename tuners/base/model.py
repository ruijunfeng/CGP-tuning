from typing import Tuple

import torch
from torch import nn

from tuners.base.gnn import GraphEncoder

class BaseGraphPromptEncoder(nn.Module):
    """
    Base class to define the graph prompt encoder for encoding graph-based structural information.
    
    Args:
        config : The configuration of the graph prompt encoder.
    """
    
    def __init__(self, config):
        super().__init__()
        # Ablation settings validity check
        if sum([config.ablate_node_type_embedding, config.ablate_edge_type_embedding, config.ablate_positional_embedding, config.ablate_cross_modal_alignment_module, config.ablate_multi_head_attn, config.ablate_projector]) > 1:
            raise ValueError(
                "Only one ablation setting can be True at a time."
            )
            
        # Graph encoder settings
        self.graph_encoder = GraphEncoder(
            input_size=config.gnn_input_size,
            hidden_size=config.gnn_hidden_size,
            output_size=config.gnn_output_size,
            num_layers=config.gnn_num_layers,
            num_heads=config.gnn_num_heads,
            dropout=config.gnn_dropout,
            bias=config.gnn_bias,
            num_node_types=config.num_node_types,
            num_edge_types=config.num_edge_types,
            max_num_nodes=config.max_num_nodes,
            edge_dim=config.edge_dim,
            ablate_node_type_embeddings=config.ablate_node_type_embeddings,
            ablate_edge_type_embeddings=config.ablate_edge_type_embeddings,
            ablate_positional_embeddings=config.ablate_positional_embeddings,
        )
        
    def reshape_node_embeds(
        self, 
        x: torch.Tensor, 
        batch_index: torch.Tensor
    )-> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reshape the node embeddings with batch axis to match the input word embeddings.

        Args:
            x (torch.Tensor): Node embeddings of shape (num_nodes, output_size).
            batch_index (torch.Tensor): Batch indices for each node (shape: [num_nodes]).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the reshaped node embeddings and the corresponding node mask.
        """
        
        # Determine the number of graphs in the batch_index and maximum number of nodes per graph
        num_graphs = batch_index.max().item() + 1
        max_num_nodes = torch.bincount(batch_index).max().item()
        output_size = x.size(1)
        
        # If there's only one graph, return x directly with a corresponding node mask
        if num_graphs == 1: 
            max_num_nodes = x.size(0)
            node_mask = torch.ones((1, max_num_nodes), dtype=torch.long, device=x.device)
            return x.unsqueeze(0), node_mask
        
        # Initialize a padded tensor to hold node embeddings for each graph
        node_embeds = torch.zeros((num_graphs, max_num_nodes, output_size), dtype=x.dtype, device=x.device)
        node_mask = torch.zeros((num_graphs, max_num_nodes), dtype=torch.long, device=x.device)
        
        # Fill in the node embeddings for each graph
        for graph_idx in range(num_graphs):
            graph_node_indices = (batch_index == graph_idx).nonzero(as_tuple=True)[0]
            num_nodes = graph_node_indices.size(0)
            node_embeds[graph_idx, :num_nodes] = x[graph_node_indices]
            node_mask[graph_idx, :num_nodes] = 1

        return node_embeds, node_mask
    