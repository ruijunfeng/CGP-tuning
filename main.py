import os
import networkx as nx

import torch
from torch_geometric.utils import from_networkx

from transformers import AutoModelForCausalLM, AutoTokenizer

from tuners.cgp_tuning.config import GraphPromptEncoderConfig
from tuners.peft_model import GraphPeftModelForCausalLM

from utils.data_utils import read_file, DataCollatorForChatML
from utils.json_utils import read_json_file, write_json_file

if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    torch.backends.mkl.enabled = False  # Disable MKL optimizations
    print("number of gpus", torch.cuda.device_count())
    print("number of cpus", os.cpu_count())

    # base model settings
    base_model_dir = "Qwen/Qwen2.5-Coder-7B" # "codellama/CodeLlama-7b-hf", "google/codegemma-7b"
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_dir)
    tokenizer.chat_template = read_file("templates/chat_template.jinja")
    tokenizer.pad_token = tokenizer.bos_token if tokenizer.pad_token is None else tokenizer.pad_token
    # base model
    base_model = AutoModelForCausalLM.from_pretrained(base_model_dir, torch_dtype="auto", trust_remote_code=True) # default model weights are in bfloat16
    base_model.gradient_checkpointing_enable() # enable gradient checkpointing for memory efficiency and long context fine-tuning
    for name, module in base_model.named_children(): # freeze all layers of the base model
        for param in module.parameters():
            param.requires_grad = False
            
    # ---------------------------------- CGP_TUNING ----------------------------------
    # Graph prompt encoder settings
    config = GraphPromptEncoderConfig(
        peft_type="CGP_TUNING",
        task_type="CAUSAL_LM",
        inference_mode=False,
        base_model_name_or_path=base_model_dir,
        token_dim=base_model.config.hidden_size, 
        # Graph encoder settings (output_size is same as the token_dim for cross-modal alignment)
        gnn_input_size=base_model.config.hidden_size,
        gnn_hidden_size=base_model.config.hidden_size,
        gnn_output_size=base_model.config.hidden_size,
        gnn_num_layers=4,
        gnn_num_heads=8,
        gnn_dropout=0.1,
        gnn_bias=True,
        num_node_types=18,
        num_edge_types=13,
        max_num_nodes=4096,
        edge_dim=128,
        ablate_node_type_embeddings=False,
        ablate_edge_type_embeddings=False,
        ablate_positional_embeddings=False,
        # Cross-modal alignment module settings
        num_virtual_tokens=32,
        cma_hidden_size=base_model.config.hidden_size, # same as token_dim
        cma_num_heads=32,
        cma_num_key_value_heads=32,
        cma_dropout=0.1,
        cma_bias=False,
        ablate_cross_modal_alignment_module=False,
        ablate_multi_head_attn=False,
        ablate_cross_modal_attention=False,
    )
    # peft model initialization
    peft_model = GraphPeftModelForCausalLM(config=config, base_model=base_model)
    peft_model.print_trainable_parameters()
    peft_model.eval()
    
    # read the graph and function from the json file
    data_sample = read_json_file("example.json")

    # process the graph
    graph = from_networkx(nx.json_graph.node_link_graph(data_sample["graph"]))
    node_type_to_index = read_json_file("templates/node_type_to_index.json")
    edge_type_to_index = read_json_file("templates/edge_type_to_index.json")
    graph.label = torch.tensor([node_type_to_index[label] for label in graph.label])
    graph.edge_label = torch.tensor([edge_type_to_index[edge_label] for edge_label in graph.edge_label])
    graph.batch = torch.zeros(graph.num_nodes, dtype=torch.int64)

    # process the input text
    message_template = read_json_file("templates/message_template.json")
    max_length = 16_000
    ignore_index = -100
    is_padding = True
    messages_key = "messages"
    examples = [
        {
            messages_key: [
                {
                    "role": "user", 
                    "content": message_template["user"].format(func=data_sample["func"])
                },
                {
                    "role": "assistant", 
                    "content": message_template["assistant"].format(target=data_sample["target"])
                },
            ]
        }
    ]
    collator = DataCollatorForChatML(
        tokenizer=tokenizer, max_length=max_length, ignore_index=ignore_index, 
        messages_key="messages", is_padding=is_padding
    )
    data = collator(examples)
    
    # forward pass
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            peft_model.to("cuda")
            outputs = peft_model(
                    input_ids=data["prompts"].to("cuda"), 
                    attention_mask=data["prompt_attention_mask"].to("cuda"), 
                    graphs=graph.to("cuda"),
                )
            
            # Constrained decoding
            logits = outputs.logits[:, -1, :] # get the last token logits
            true_token_id = tokenizer.convert_tokens_to_ids("true") # get the token id for true
            false_token_id = tokenizer.convert_tokens_to_ids("false") # get the token id for false
            results = logits[:, true_token_id] > logits[:, false_token_id] # get the y_pred