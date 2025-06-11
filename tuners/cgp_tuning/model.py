from typing import Union, List

import torch
from torch import nn
import torch_geometric

from transformers.modeling_outputs import ModelOutput

from tuners.base.attn import MultiHeadAttention
from tuners.base.model import BaseGraphPromptEncoder
from tuners.cgp_tuning.config import GraphPromptEncoderConfig

class DummyLinear(nn.Module):
    """
    A dummy linear layer that does not perform any operation.
    This is used to ablate the projector in the GraphPromptEncoder.
    It initializes the weight and bias parameters but does not apply any transformation in the forward pass.
    
    Args:
        in_features (int): The number of input features.
        out_features (int): The number of output features.
        bias (bool, optional): Whether to include a bias term. Defaults to True.
    """
    
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        bias: bool = True
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input
    
class GraphPromptEncoder(BaseGraphPromptEncoder):
    """
    Use the graph encoder to extract graph features, then use shared cross-modal alignment module, to align virtual tokens with graph features and text features.
    
    Args:
        config ([`GraphPromptEncoderConfig`]): The configuration of the cross_modal_alignment_module.
    """
    
    def __init__(self, config: GraphPromptEncoderConfig):
        super().__init__(config)
        # Initialize the virtual token embeddings
        self.embedding = nn.Embedding(config.num_virtual_tokens, config.token_dim).weight
        
        # Initialize the cross-modal alignment module
        self.multi_head_attn = MultiHeadAttention(
            hidden_size=config.cma_hidden_size,
            num_heads=config.cma_num_heads,
            num_key_value_heads=config.cma_num_key_value_heads,
            dropout=config.cma_dropout,
            bias=config.cma_bias,
        )
        self.projector = nn.Linear(config.cma_hidden_size, config.cma_hidden_size, bias=True)
        
        # Ablation of cross-modal alignment module, multi-head attention, or projector
        if config.ablate_cross_modal_alignment_module:
            self.forward = lambda graphs=None, input_embeds=None, attention_mask=None, **kwargs: ModelOutput(soft_prompts=self.embedding.expand(input_embeds.size(0), -1, -1))
        if config.ablate_multi_head_attn:
            self.forward = lambda graphs=None, input_embeds=None, attention_mask=None, **kwargs: ModelOutput(soft_prompts=self.projector(self.embedding.expand(input_embeds.size(0), -1, -1)))
        if config.ablate_projector:
            self.projector = DummyLinear(config.cma_hidden_size, config.cma_hidden_size, bias=True)
    
    def forward(
        self, 
        graphs: torch_geometric.data.Data = None,
        input_embeds: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        **kwargs,
    ) -> Union[List, ModelOutput]:
        """
        Forward pass with cross-modal alignment and projection.
        
        Args:
            graphs (torch_geometric.data.Data, optional): graph data object. Defaults to None.
            input_embeds (torch.Tensor, optional): input word embeddings. Defaults to None.
            attention_mask (torch.Tensor, optional): attention mask for the input word embeddings. Defaults to None.
        
        Returns:
            Union[List, ModelOutput]: A list containing the soft prompts, node embeddings, and node mask.
        """
        # Extract the graph features (output_size = token_dim so no projection is needed)
        x, batch_index = self.graph_encoder(graphs).values() # Shape: (batch_size * num_nodes, output_size)
        
        # Reshape the node embeddings to match batch format
        node_embeds, node_mask = self.reshape_node_embeds(x, batch_index) # Shape: (batch_size, num_nodes, output_size) and (batch_size, num_nodes)
        
        # Expand the soft prompt embeddings along the batch axis to match the shape of input word embeddings
        soft_prompt_embeds = self.embedding.expand(input_embeds.size(0), -1, -1) # Shape: (batch_size, num_virtual_tokens, token_dim)
        
        # Multi-head attention
        text_features = self.multi_head_attn(query=soft_prompt_embeds, key=input_embeds, value=input_embeds, key_padding_mask=attention_mask) # Shape: (batch_size, num_virtual_tokens, token_dim)
        graph_features = self.multi_head_attn(query=text_features, key=node_embeds, value=node_embeds, key_padding_mask=node_mask) # Shape: (batch_size, num_virtual_tokens, token_dim)
        
        # Projector
        soft_prompts = self.projector(graph_features)
        
        return ModelOutput(
            soft_prompts=soft_prompts,
            node_embeds=node_embeds,
            node_mask=node_mask,
        )
