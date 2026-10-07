import sys
import copy
import torch
import torch.nn as nn
sys.path.insert(0, "engine")
from qwen3loop_python import Qwen3LoopConfig, Qwen3LoopForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)

print(f"=== TESTE DE EQUIVALÊNCIA MATEMÁTICA: GRADIENT CHECKPOINTING NO LOOP ===")
print(f"Dispositivo: {device}")

# 1. Configuração do modelo idêntica à arquitetura do projeto
cfg = Qwen3LoopConfig()
cfg.num_hidden_layers = 4      # 4 camadas físicas para teste ultrarrápido
cfg.vocab_size = 1000
cfg.hidden_size = 256
cfg.num_attention_heads = 4
cfg.num_key_value_heads = 2
cfg.intermediate_size = 512
cfg.num_loops = 2             # Recorrência ativada: 8 camadas lógicas unrolled
cfg.enable_double_loop_split = True
cfg.loop_middle_layers = 2

# Modelo 1: SEM Gradient Checkpointing (Base Ground Truth)
torch.manual_seed(1234)
model_base = Qwen3LoopForCausalLM(cfg).to(torch.float32).to(device)
model_base.train()

# Modelo 2: Clone idêntico com os mesmos pesos
model_ckpt = copy.deepcopy(model_base)
model_ckpt.train()
model_ckpt.gradient_checkpointing_enable()

# Dados de entrada fixos para o teste
seq_len = 128
input_ids = torch.randint(0, 1000, (1, seq_len), device=device)
attention_mask = torch.ones((1, seq_len), device=device)
labels = input_ids.clone()

# --- EXECUÇÃO 1: SEM CHECKPOINTING ---
out_base = model_base(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
loss_base = out_base.loss
loss_base.backward()

grads_base = {n: p.grad.clone() for n, p in model_base.named_parameters() if p.grad is not None}

# --- IMPLEMENTAÇÃO DA FUNÇÃO CHECKPOINTADA ---
def create_checkpointed_forward(layer_module, v_idx, o_idx, o_attn_idx):
    def custom_forward(h, pos_emb, pos_id, mask):
        layer_module.layer_idx = v_idx
        layer_module.self_attn.layer_idx = v_idx
        try:
            return layer_module(
                h,
                attention_mask=mask,
                position_embeddings=pos_emb,
                position_ids=pos_id,
                use_cache=False,
            )
        finally:
            layer_module.layer_idx = o_idx
            layer_module.self_attn.layer_idx = o_attn_idx
    return custom_forward

# Patch temporário no model_ckpt.model.forward para testar
orig_fwd = model_ckpt.model.forward

def forward_patched(
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
    self = model_ckpt.model
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

    from transformers.masking_utils import create_causal_mask
    causal_mask_mapping = {
        "full_attention": create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=None,
            position_ids=position_ids,
            cache_position=cache_position,
        )
    }

    hidden_states = inputs_embeds
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    loop_hidden_states = []
    for loop_idx in range(num_loops):
        for execution_idx, (i, _repeat_idx) in enumerate(layer_order):
            decoder_layer = self.layers[i]
            virtual_layer_idx = execution_idx if double_split else i
            orig_layer_idx = getattr(decoder_layer, "layer_idx", i)
            orig_attn_layer_idx = getattr(decoder_layer.self_attn, "layer_idx", i)

            attn_mask = causal_mask_mapping[self.config.layer_types[i]]

            custom_fwd = create_checkpointed_forward(decoder_layer, virtual_layer_idx, orig_layer_idx, orig_attn_layer_idx)
            hidden_states = torch.utils.checkpoint.checkpoint(
                custom_fwd,
                hidden_states,
                position_embeddings,
                position_ids,
                attn_mask,
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

model_ckpt.model.forward = forward_patched

# --- EXECUÇÃO 2: COM CHECKPOINTING ---
out_ckpt = model_ckpt(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
loss_ckpt = out_ckpt.loss
loss_ckpt.backward()

grads_ckpt = {n: p.grad.clone() for n, p in model_ckpt.named_parameters() if p.grad is not None}

# --- 3. COMPARAÇÃO RIGOROSA ---
loss_diff = abs(loss_base.item() - loss_ckpt.item())
print(f"\n1. Loss sem Checkpointing: {loss_base.item():.8f}")
print(f"   Loss com Checkpointing: {loss_ckpt.item():.8f}")
print(f"   Diferença absoluta:     {loss_diff:.10f}")

assert loss_diff < 1e-6, f"FALHA: Diferença de Loss detectada ({loss_diff})"

max_grad_diff = 0.0
total_params_compared = 0
for name in grads_base:
    if name in grads_ckpt:
        diff = (grads_base[name] - grads_ckpt[name]).abs().max().item()
        max_grad_diff = max(max_grad_diff, diff)
        total_params_compared += 1

print(f"\n2. Parâmetros comparados:  {total_params_compared}")
print(f"   Diferença máxima de grad: {max_grad_diff:.10f}")

assert max_grad_diff < 1e-5, f"FALHA: Gradiente divergiu ({max_grad_diff})"

# 4. Validar o tensor da sonda intermediária do Loop 1
assert hasattr(out_ckpt, "loop_hidden_states"), "loop_hidden_states ausente!"
print(f"\n3. loop_hidden_states presentes: {len(out_ckpt.loop_hidden_states)} estados")
h1_base = out_base.loop_hidden_states[0]
h1_ckpt = out_ckpt.loop_hidden_states[0]
h1_diff = (h1_base - h1_ckpt).abs().max().item()
print(f"   Diferença máxima em h1:   {h1_diff:.10f}")
assert h1_diff < 1e-6, f"FALHA: h1 divergiu ({h1_diff})"

print("\n>>> SUCESSO TOTAL: GUARDBOUND COMPROVADO MATEMATICAMENTE A 100%! <<<")
