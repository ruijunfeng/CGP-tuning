from typing import Optional

import torch
from torch import nn

from torch_geometric.data import Data
from tuners.cgp_tuning.model import GraphPromptEncoder

class GraphPeftModelForCausalLM(nn.Module):
    """
    Similar to PeftModelForCausalLM, this class is a wrapper for graph-enhanced soft prompt tuning. 
    The difference is that the prompt encoder here takes a graph as input.
    
    Args:
        config: The configuration of the prompt encoder.
        base_model: The base model to be used.
    """
    def __init__(self, config, base_model):
        super().__init__()
        self.config = config
        self.prompt_encoder = GraphPromptEncoder(config)
        self.base_model = base_model
        self.word_embeddings = self.base_model.get_input_embeddings()
    
    def forward(
        self, 
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        labels: Optional[torch.Tensor] = None,
        graphs: Data = None,
        **kwargs,
    ):
        # Initialize the input_embeds
        batch_size = input_ids.shape[0]
        input_embeds = self.word_embeddings(input_ids)
        
        # Generate soft prompts
        if labels is not None: # find the label part in the input_ids
            label_mask = (labels == -100).int().to(device=input_ids.device, dtype=torch.long)
        else: # if labels is None, then keep the whole input_ids
            label_mask = torch.ones(attention_mask.shape, dtype=torch.long, device=input_ids.device)
        prompt_outputs = self.get_prompt(
            graphs=graphs,
            input_embeds=input_embeds,
            attention_mask=attention_mask * label_mask, # mask the labels part in the input_ids for GNP and CGP_Tuning
        )
        total_virtual_tokens = prompt_outputs["soft_prompts"].size(1)
        
        # Concat soft_prompts with input_embeds
        input_embeds = torch.cat((prompt_outputs["soft_prompts"], input_embeds), dim=1)
        # Concat attention mask with prefix
        if attention_mask is not None:
            prefix_attention_mask = torch.ones(batch_size, total_virtual_tokens).to(attention_mask.device)
            attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
        # Concat labels with prefix
        if labels is not None:
            prefix_labels = torch.full((batch_size, total_virtual_tokens), -100).to(labels.device) # prefix the labels with -100 (ignore index)
            labels = torch.cat((prefix_labels, labels), dim=1)
        
        # Forward pass with the base_model
        outputs = self.base_model(
            inputs_embeds=input_embeds, 
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )
        
        # Adds the prompt_outputs to the outputs
        for key, value in prompt_outputs.items():
            outputs[key] = value
            
        return outputs
    
    def print_trainable_parameters(self):
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        # Count total parameters
        total_params = sum(p.numel() for p in self.parameters())
        # Calculate trainable percentage
        trainable_percent = trainable_params / total_params * 100 if total_params > 0 else 0
        # Print the information in the desired format
        print(f"trainable params: {trainable_params:,} || all params: {total_params:,} || trainable%: {trainable_percent:.4f}")
    
    def get_prompt(self, **kwargs):
        return self.prompt_encoder(**kwargs)
    