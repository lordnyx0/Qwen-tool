import sys
sys.path.insert(0, ".")
import torch
from torch.utils.checkpoint import checkpoint
import qwen3loop_python
from qwen3loop_python import Qwen3LoopConfig, Qwen3LoopForCausalLM

cfg = Qwen3LoopConfig.from_pretrained('checkpoints/qwen3loop_sft_final')
cfg.loop_loss_weights = []
model = Qwen3LoopForCausalLM(cfg).to(torch.bfloat16).cuda()
model.train()
model.gradient_checkpointing_enable()

print(f"Model initialized on CUDA. Weights VRAM: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

# Test custom forward with checkpointing
orig_model_forward = model.model.forward

def forward_with_checkpointing(
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    cache_position=None,
    **kwargs,
):
    self = model.model
    num_loops = self.get_num_loops()
    layer_order = self.get_layer_execution_order()
    double_split = getattr(self.config, "enable_double_loop_split", False)
    use_cache = False

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    past_seen_tokens = 0
    cache_position = torch.arange(0, inputs_embeds.shape[1], device=inputs_embeds.device)
    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    # In PyTorch SDPA, causal mask can be None if is_causal is True
    hidden_states = inputs_embeds
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    loop_hidden_states = []
    for loop_idx in range(num_loops):
        for execution_idx, (i, _repeat_idx) in enumerate(layer_order):
            decoder_layer = self.layers[i]
            
            def create_custom_forward(module):
                def custom_forward(h, pos_emb, pos_id):
                    return module(h, attention_mask=None, position_embeddings=pos_emb, position_ids=pos_id, use_cache=False)
                return custom_forward

            hidden_states = checkpoint(
                create_custom_forward(decoder_layer),
                hidden_states,
                position_embeddings,
                position_ids,
                use_reentrant=False
            )

            if double_split and _repeat_idx is not None:
                if execution_idx + 1 < len(layer_order) and layer_order[execution_idx + 1][1] != _repeat_idx:
                    loop_hidden_states.append(hidden_states)

    hidden_states = self.norm(hidden_states)
    from transformers.modeling_outputs import BaseModelOutputWithPast
    out = BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=None,
        hidden_states=None,
    )
    out.loop_hidden_states = tuple(loop_hidden_states)
    return out

model.model.forward = forward_with_checkpointing

seq_len = 8192
input_ids = torch.randint(0, 1000, (1, seq_len), device='cuda')
labels = input_ids.clone()

print(f"Starting 8K forward + backward with REAL gradient checkpointing...")
with torch.autocast('cuda', dtype=torch.bfloat16):
    out = model(input_ids=input_ids, labels=labels)
    loss = out.loss
    print(f"Forward OK! Allocated VRAM: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    loss.backward()
    print(f"Backward OK! Peak VRAM: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

last_grad = model.model.layers[27].self_attn.q_proj.weight.grad
print(f"Grad on layer 27 exists: {last_grad is not None}, norm: {last_grad.abs().sum().item():.2f}")
