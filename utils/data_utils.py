from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import torch
from trl.trainer.utils import pad
from transformers import PreTrainedTokenizerBase

def read_file(file_path: str) -> str:
    with open(file_path, "r") as file:
        data = file.read()
    return data

# Updates the incorrect data processing in the original implementation
# Reference: https://github.com/huggingface/trl/issues/2169.
# Customize to not set a reasonable value for max_length if it is None, and do not return padded sequences when max_length is None for testing
@dataclass
class DataCollatorForChatML:
    """
    Data collator for ChatML format datasets.
    """

    tokenizer: PreTrainedTokenizerBase
    ignore_index: int = -100
    max_length: int = None
    messages_key: str = "messages"
    is_padding: Optional[bool] = True
    
    def __post_init__(self):
        if self.tokenizer.pad_token_id is None:
            raise ValueError("The tokenizer does not have a pad token. Please set `pad_token_id` in the tokenizer.")
        
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        prompts = []
        completions = []
        
        for example in examples:
            messages = example[self.messages_key]
            formatted_chat = self.tokenizer.apply_chat_template(messages, tokenize=False)
            
            # Split the formatted chat into prompt and completion
            assistant_messages = [msg for msg in messages if msg["role"] == "assistant"]
            last_assistant_message = assistant_messages[-1]["content"]
            prompt = formatted_chat.rsplit(last_assistant_message, 1)[0]
            completion = last_assistant_message + formatted_chat.rsplit(last_assistant_message, 1)[1]
            
            prompts.append(prompt)
            completions.append(completion)
            
        # Tokenize prompts and completions
        tokenized_prompts = self.tokenizer(
            prompts,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
            # We assume the inputs are already wrapped with BOS&EOS tokens in tokenizer.apply_chat_template, so extra BOS/EOS tokens should not be added
            add_special_tokens=False,
        )
        tokenized_completions = self.tokenizer(
            completions,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
            # We assume the inputs are already wrapped with BOS&EOS tokens in tokenizer.apply_chat_template, so extra BOS/EOS tokens should not be added
            add_special_tokens=False,
        )
        
        # Combine prompts and completions
        input_ids = []
        attention_mask = []
        labels = []
        
        for prompt, completion in zip(tokenized_prompts["input_ids"], tokenized_completions["input_ids"]):
            combined_input_ids = prompt + completion
            combined_attention_mask = [1] * len(combined_input_ids)
            
            # Create labels for one-token ahead task, masking the prompt
            combined_labels = [self.ignore_index] * len(prompt) + completion[:-1]
            combined_labels.append(self.tokenizer.eos_token_id)  # Add EOS token as final target
            
            input_ids.append(combined_input_ids)
            attention_mask.append(combined_attention_mask)
            labels.append(combined_labels)
            
        # first convert to list of tensors
        input_ids = [torch.tensor(ids) for ids in input_ids]
        attention_mask = [torch.tensor(mask) for mask in attention_mask]
        labels = [torch.tensor(label) for label in labels]
        prompts_input_ids = [torch.tensor(ids) for ids in tokenized_prompts["input_ids"]]
        prompt_attention_mask = [torch.tensor([1] * len(ids)) for ids in tokenized_prompts["input_ids"]]
        
        # pad the input_ids, attention_mask and labels to the same length across the batch if max_length is set
        if self.is_padding:
            input_ids = pad(input_ids, padding_side="left", padding_value=self.tokenizer.pad_token_id)
            attention_mask = pad(attention_mask, padding_side="left", padding_value=0)
            labels = pad(labels, padding_side="left", padding_value=self.ignore_index)
            prompts_input_ids = pad(prompts_input_ids, padding_side="left", padding_value=self.tokenizer.pad_token_id)
            prompt_attention_mask = pad(prompt_attention_mask, padding_side="left", padding_value=0)
            
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "prompts": prompts_input_ids,
            "prompt_attention_mask": prompt_attention_mask,
        }