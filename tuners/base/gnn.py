from typing import Union, List

import torch
from torch import nn
from torch.nn import functional as F

import torch_geometric
from torch_geometric.nn import GATConv, ASAPooling, LayerNorm

from transformers.modeling_outputs import ModelOutput

class GraphEncoder(nn.Module):
    """
    Graph encoder to encode the graph-based structural infomation as soft prompt embeddings.
    
    Args:
        input_size (int): The input dimension of node embeddings for the input layer.
        output_size (int): The output dimension of node embeddings for the output layer.
        hidden_size (int): The hidden dimension of node embeddings for the hidden layers.
        num_layers (int): The number of layers (including input, hidden, and output).
        num_heads (int): The number of attention heads in layers.
        dropout (float): The dropout probability for the attention weights in layers.
        bias (bool): Whether to include bias in the output of layers.
        num_node_types (int): The number of node types in the graph.
        num_edge_types (int): The number of edge types in the graph.
        max_num_nodes (int): The maximum number of nodes in the graph.
        edge_dim (int): The dimension of edge embeddings.
    """
    
    def __init__(
        self, 
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        bias: bool,
        num_node_types: int,
        num_edge_types: int,
        max_num_nodes: int,
        edge_dim: int,
        ablate_node_type_embeddings: bool,
        ablate_edge_type_embeddings: bool,
        ablate_positional_embeddings: bool,
    ):
        super().__init__()
        # Initialize the settings
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.bias = bias
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types
        self.max_num_nodes = max_num_nodes
        self.edge_dim = edge_dim
        
        # Initialize the node type embeddings
        self.node_type_embeddings = nn.Embedding(self.num_node_types, self.input_size)
        
        # Initialize the edge type embeddings
        self.edge_type_embeddings = nn.Embedding(self.num_edge_types, self.edge_dim)
        
        # Ablation of node type embeddings, edge type embeddings, or positional embeddings
        if ablate_node_type_embeddings:
            self.generate_node_embeddings = self.generate_sinusoidal_positional_embeddings
        elif ablate_edge_type_embeddings:
            self.edge_dim = None
            self.generate_edge_embeddings = lambda *args, **kwargs: None
        elif ablate_positional_embeddings:
            self.generate_node_embeddings = lambda batch_index, d_modal, label: self.node_type_embeddings(label)
        
        # Initialize the graph layers (no activation function is needed between GATConv as it is already included in the GATConv)
        # This implementation simulates the multi-head attention by setting concat=True across the GATConv layers
        # ModuleList to hold the GATConv layers
        self.graph_layers = nn.ModuleList()
        self.graph_norms = nn.ModuleList()
        
        # Input layer
        self.graph_layers.append(
            GATConv(
                self.input_size, self.hidden_size // self.num_heads, edge_dim=self.edge_dim,
                heads=self.num_heads, dropout=0.0, concat=True, bias=self.bias
            )
        )
        self.graph_norms.append(LayerNorm(self.hidden_size))
        
        # Hidden layers
        if self.num_layers > 2:
            for i in range(self.num_layers - 2):
                self.graph_layers.append(
                    GATConv(
                        self.hidden_size, self.hidden_size // self.num_heads, edge_dim=self.edge_dim, 
                        heads=self.num_heads, dropout=0.0, concat=True, bias=self.bias
                    )
                )
                self.graph_norms.append(LayerNorm(self.hidden_size))
        
        # Output layer
        self.graph_layers.append(
            GATConv(
                self.hidden_size, self.output_size // self.num_heads, edge_dim=self.edge_dim,
                heads=self.num_heads, dropout=0.0, concat=True, bias=self.bias
            )
        )
        self.graph_norms.append(LayerNorm(self.output_size))
        
        # Pooling layer
        self.pooling = ASAPooling(self.output_size, ratio=self.max_num_nodes)
        
    def generate_sinusoidal_positional_embeddings(
        self,
        batch_index: torch.Tensor,
        d_model: int,
        *args,
    ) -> torch.Tensor:
        """
        Create sinusoidal positional embeddings for a batched graph.
        
        Args:
            batch_index (torch.Tensor): A tensor containing batch indices for each node (shape: [num_nodes]).
            d_model (int): Dimension of the initial node embeddings.
        
        Returns:
            torch.Tensor: A tensor of shape (num_nodes, d_model) containing the sinusoidal positional embeddings.
        """
        
        def generate_sinusoidal_positional_embedding( 
            num_nodes: int, 
            d_model: int,
            device: torch.device
        ) -> torch.Tensor:
            """
            Create a sinusoidal positional encoding matrix.
            
            Args:
                num_nodes (int): Number of nodes in the graph.
                d_model (int): Dimension of the initial node embeddings.
                device (torch.device): Device to put the positional embeddings.
                
            Returns:
                torch.Tensor: A tensor of shape (num_nodes, d_model) containing the positional embeddings.
            """
            # Create a matrix of shape (num_nodes, 1) containing positions (0, 1, ..., num_nodes-1)
            position = torch.arange(num_nodes, dtype=torch.float).unsqueeze(1)  # Shape: (num_nodes, 1)
            
            # Create a matrix of shape (1, d_model) for div_term
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))  # Shape: (d_model / 2)
            
            # Initialize the positional encoding matrix
            position_embed = torch.zeros(num_nodes, d_model, device=device)  # Shape: (num_nodes, d_model)
            
            # Apply sine to even indices and cosine to odd indices
            position_embed[:, 0::2] = torch.sin(position * div_term)  # Even indices: sin
            position_embed[:, 1::2] = torch.cos(position * div_term[: d_model // 2])  # Odd indices: cos (d_model // 2 to match the size if input size is odd)
            
            return position_embed # Shape: (num_nodes, d_model)
        
        # Count the number of nodes per graph in the batch
        node_counts = torch.bincount(batch_index)  # Shape: (num_graphs,)
        
        # Generate positional embeddings for each graph separately
        position_embeds = []
        for num_nodes in node_counts:
            # Generate positional encoding for the current graph
            position_embed = generate_sinusoidal_positional_embedding(num_nodes.item(), d_model, device=batch_index.device)
            position_embeds.append(position_embed)
        
        # Concatenate all positional embeddings
        return torch.cat(position_embeds, dim=0)  # Shape: (batch_size * num_nodes, d_model)
    
    def generate_node_embeddings(
        self,
        batch_index: torch.Tensor,
        d_modal: int,
        label: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate node embeddings for the input graph data.
        
        Args:
            batch_index (torch.Tensor): A tensor containing batch indices for each node (shape: [num_nodes]).
            d_model (int): Dimension of the initial node embeddings.
            label (torch.Tensor): A tensor containing node type labels for each node (shape: [num_nodes]).
            
        Returns:
            torch.Tensor: A tensor of shape (batch_size * num_nodes, input_size) containing the node embeddings.
        """
        # Prepare the initial node embeddings
        position_embeds = self.generate_sinusoidal_positional_embeddings(batch_index, d_modal)
        node_type_embeds = self.node_type_embeddings(label)
        x = position_embeds + node_type_embeds
        return x
    
    def generate_edge_embeddings(
        self,
        edge_label: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate edge embeddings for the input graph data.
        
        Args:
            edge_label (torch.Tensor): A tensor containing edge type labels for each edge (shape: [num_edges]).
            
        Returns:
            torch.Tensor: A tensor of shape (num_edges, edge_dim) containing the edge embeddings.
        """
        edge_type_embeds = self.edge_type_embeddings(edge_label)
        return edge_type_embeds
    
    def forward(
        self, 
        graphs: torch_geometric.data.Data = None,
    ) -> Union[List, ModelOutput]:
        """
        Forward pass of the graph prompt encoder network.
        
        Args:
            graphs (torch_geometric.data.Data, optional): graph data object. Defaults to None.

        Returns:
            Union[List, ModelOutput]: A list containing the node embeddings and batch index.
        """
        # Prepare the graph features
        label = graphs.label
        edge_index = graphs.edge_index
        edge_label = graphs.edge_label
        batch_index = graphs.batch
        x = self.generate_node_embeddings(batch_index, self.input_size, label) # Shape: (batch_size * num_nodes, input_size)
        edge_attr = self.generate_edge_embeddings(edge_label) # Shape: (num_edges, edge_dim)
        
        # Encode edge features into node embeddings
        for i, layer in enumerate(self.graph_layers):
            residual = x
            x = layer(x, edge_index, edge_attr)
            x = self.graph_norms[i](x, batch_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + residual
            
        # Pooling layer to aggregate node embeddings
        x, edge_index, _, batch_index, _ = self.pooling(x=x, edge_index=edge_index, batch=batch_index)  # Pooling layer modifies the graph structure         
        
        return ModelOutput(
            x=x,
            batch=batch_index,
        )
        