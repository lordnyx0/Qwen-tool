# coding=utf-8
# Copyright 2026 Qwen3Loop team and HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0 (the "License");

""" Qwen3Loop Model Architecture - Looped Transformer for Qwen3 via Direct HF Inheritance

Paridade de arquitetura com o Nanbeige4.2-3B (modeling_nanbeige.py). O laço de
loop, a indexação virtual do KV cache e a norm entre passagens seguem o mesmo
esquema; os defaults reproduzem os pesos publicados do Nanbeige4.2-3B, que NÃO
ativam LoopSplit, mHC, depth attention nem n-gram embeddings.
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Model,
    Qwen3ForCausalLM,
    create_causal_mask,
    create_sliding_window_causal_mask,
)
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from .configuration_qwen3loop import Qwen3LoopConfig


def get_loop_cache_layer_idx(layer_idx: int, loop_idx: int, num_hidden_layers: int,
                             cache_layer_idx: Optional[int] = None) -> int:
    """Índice virtual de camada no KV cache. Igual a _get_loop_cache_layer_idx do Nanbeige."""
    if cache_layer_idx is not None:
        return cache_layer_idx
    return layer_idx + loop_idx * num_hidden_layers


def get_double_loop_split_layer_order(num_hidden_layers: int,
                                      loop_middle_layers: Optional[int] = None
                                      ) -> List[Tuple[int, Optional[int]]]:
    """
    LoopSplit. Porte fiel de _get_double_loop_split_layer_order_with_mhc_loop_indices.

    Mantém um prefixo e um sufixo executados uma única vez e repete só o miolo.
    Para N=22, M=11: [0..4] uma vez, [5..15] tres vezes, [16..21] uma vez = 44
    execuções — mesmo custo de 2 loops completos, distribuído de outra forma.

    Devolve pares (layer_idx, repeat_idx), com repeat_idx=None nas camadas não iteradas.
    """
    if num_hidden_layers <= 0:
        raise ValueError("enable_double_loop_split requer num_hidden_layers > 0.")
    if loop_middle_layers is None:
        if num_hidden_layers % 2 != 0:
            raise ValueError(
                "enable_double_loop_split requer num_hidden_layers divisível por 2 "
                "quando loop_middle_layers não é definido."
            )
        loop_middle_layers = num_hidden_layers // 2
    if loop_middle_layers <= 0:
        raise ValueError("loop_middle_layers deve ser > 0.")
    if num_hidden_layers % loop_middle_layers != 0:
        raise ValueError("loop_middle_layers deve ser divisor de num_hidden_layers.")

    first_unlooped = (num_hidden_layers - loop_middle_layers) // 2
    middle_start = first_unlooped
    middle_end = middle_start + loop_middle_layers
    middle_repeats = (num_hidden_layers + loop_middle_layers) // loop_middle_layers
    return (
        [(idx, None) for idx in range(0, middle_start)]
        + [
            (idx, repeat_idx)
            for repeat_idx in range(middle_repeats)
            for idx in range(middle_start, middle_end)
        ]
        + [(idx, None) for idx in range(middle_end, num_hidden_layers)]
    )


class Qwen3LoopModel(Qwen3Model):
    """
    Qwen3LoopModel inherits directly from HF's Qwen3Model.
    Reuses all native Qwen3 modules (Qwen3Attention, Qwen3DecoderLayer, Qwen3RMSNorm, Qwen3MLP)
    and overrides forward() to execute the layer stack num_loops times with loop-virtual KV cache indexing.
    """
    config_class = Qwen3LoopConfig

    # ------------------------------------------------------------------
    # Helpers de loop (espelham NanbeigeModel)
    # ------------------------------------------------------------------
    def get_num_loops(self) -> int:
        if getattr(self.config, "enable_double_loop_split", False):
            return 1
        weights = getattr(self.config, "loop_loss_weights", None)
        if weights:
            return len(weights) + 1
        return getattr(self.config, "num_loops", 1)

    def get_layer_execution_order(self) -> List[Tuple[int, Optional[int]]]:
        if getattr(self.config, "enable_double_loop_split", False):
            return get_double_loop_split_layer_order(
                self.config.num_hidden_layers,
                getattr(self.config, "loop_middle_layers", None),
            )
        return [(i, None) for i in range(self.config.num_hidden_layers)]

    def get_cache_seq_length(self, past_key_values: Optional[Cache]) -> int:
        """
        O comprimento do cache tem que sair de uma camada que de fato escreveu nele.
        Com LoopSplit é a camada 0; caso contrário, o máximo entre as camadas
        virtuais que iniciam cada passagem. Espelha _get_cache_seq_length do Nanbeige.
        """
        if past_key_values is None:
            return 0
        if getattr(self.config, "enable_double_loop_split", False):
            return past_key_values.get_seq_length(0)
        max_len = 0
        for loop_idx in range(self.get_num_loops()):
            max_len = max(max_len, past_key_values.get_seq_length(loop_idx * self.config.num_hidden_layers))
        return max_len

    def _total_virtual_layers(self) -> int:
        if getattr(self.config, "enable_double_loop_split", False):
            return len(self.get_layer_execution_order())
        return self.get_num_loops() * self.config.num_hidden_layers

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        num_loops = self.get_num_loops()
        layer_order = self.get_layer_execution_order()
        double_split = getattr(self.config, "enable_double_loop_split", False)
        share_kv = getattr(self.config, "loop_share_kv", False)

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Gradient checkpointing e cache são incompatíveis (o cache seria reescrito
        # na recomputação do backward).
        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        if share_kv:
            # Compartilhar o KV da 1a passagem exige interceptar o forward da
            # atenção (o Nanbeige usa _apply_loop_shared_kv dentro do NanbeigeAttention).
            # Aqui reusamos a Qwen3Attention nativa do HF, que não tem esse gancho,
            # então a opção falha alto em vez de silenciosamente não fazer nada.
            # O relatório do Nanbeige4.2 mede ganho consistentemente menor com KV
            # compartilhado, então isto não bloqueia a paridade com os pesos publicados.
            raise NotImplementedError(
                "loop_share_kv=True exige uma subclasse de Qwen3Attention; não implementado. "
                "Os pesos publicados do Nanbeige4.2-3B usam loop_share_kv=False."
            )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache:
            if past_key_values is None:
                past_key_values = DynamicCache(config=self.config)
            target_virtual_layers = self._total_virtual_layers()
            if hasattr(past_key_values, "layers"):
                while len(past_key_values.layers) < target_virtual_layers:
                    layer_cls = getattr(past_key_values, "layer_class_to_replicate", None)
                    if layer_cls is None:
                        from transformers.cache_utils import DynamicLayer
                        layer_cls = DynamicLayer
                    past_key_values.layers.append(layer_cls())

        past_seen_tokens = self.get_cache_seq_length(past_key_values)
        cache_position = torch.arange(past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # Prepare causal mask
        # If attention_mask is None or has no padding under SDPA, bypass creating 4D float mask
        # so PyTorch SDPA uses fast implicit causal Cutlass kernel with is_causal=True!
        use_implicit_causal = False
        if not getattr(self, "has_sliding_layers", False):
            if attention_mask is None:
                use_implicit_causal = True
            elif isinstance(attention_mask, torch.Tensor) and attention_mask.dim() == 2 and (attention_mask == 1).all():
                use_implicit_causal = True

        if use_implicit_causal:
            causal_mask_mapping = {l_type: None for l_type in self.config.layer_types}
        elif not isinstance(causal_mask_mapping := attention_mask, dict):
            import inspect
            causal_params = set(inspect.signature(create_causal_mask).parameters.keys())
            embeds_key = "inputs_embeds" if "inputs_embeds" in causal_params else "input_embeds"
            mask_kwargs = {
                "config": self.config,
                embeds_key: inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
                "cache_position": cache_position,
            }
            filtered_kwargs = {k: v for k, v in mask_kwargs.items() if k in causal_params}
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**filtered_kwargs),
            }
            if getattr(self, "has_sliding_layers", False):
                sliding_params = set(inspect.signature(create_sliding_window_causal_mask).parameters.keys())
                filtered_sliding = {k: v for k, v in mask_kwargs.items() if k in sliding_params}
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**filtered_sliding)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None
        loop_hidden_states = []
        for loop_idx in range(num_loops):
            for execution_idx, (i, _repeat_idx) in enumerate(layer_order):
                decoder_layer = self.layers[i]
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                # Índice virtual para o KV cache. Com LoopSplit a posição na ordem de
                # execução é que define o slot; sem ele, o deslocamento por passagem.
                if double_split:
                    virtual_layer_idx = execution_idx
                else:
                    virtual_layer_idx = get_loop_cache_layer_idx(i, loop_idx, self.config.num_hidden_layers)

                orig_layer_idx = getattr(decoder_layer, "layer_idx", i)
                orig_attn_layer_idx = getattr(decoder_layer.self_attn, "layer_idx", i)

                attn_mask = causal_mask_mapping[self.config.layer_types[i]]

                if self.gradient_checkpointing and self.training:
                    def create_checkpointed_layer(layer_module, v_idx, o_idx, o_attn_idx):
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

                    custom_fwd = create_checkpointed_layer(decoder_layer, virtual_layer_idx, orig_layer_idx, orig_attn_layer_idx)
                    hidden_states = checkpoint(
                        custom_fwd,
                        hidden_states,
                        position_embeddings,
                        position_ids,
                        attn_mask,
                        use_reentrant=False
                    )
                else:
                    decoder_layer.layer_idx = virtual_layer_idx
                    decoder_layer.self_attn.layer_idx = virtual_layer_idx
                    try:
                        hidden_states = decoder_layer(
                            hidden_states,
                            attention_mask=attn_mask,
                            position_embeddings=position_embeddings,
                            position_ids=position_ids,
                            past_key_values=past_key_values,
                            use_cache=use_cache,
                            **kwargs,
                        )
                    finally:
                        decoder_layer.layer_idx = orig_layer_idx
                        decoder_layer.self_attn.layer_idx = orig_attn_layer_idx

                if double_split and _repeat_idx is not None:
                    if execution_idx + 1 < len(layer_order) and layer_order[execution_idx + 1][1] != _repeat_idx:
                        loop_hidden_states.append(hidden_states)

            # Norm ao fim de cada passagem (skip_loop_final_norm=False, como no
            # Nanbeige4.2 publicado). Com True, só uma vez ao final de tudo.
            if not getattr(self.config, "skip_loop_final_norm", False):
                hidden_states = self.norm(hidden_states)

            if not double_split:
                loop_hidden_states.append(hidden_states)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        if getattr(self.config, "skip_loop_final_norm", False):
            hidden_states = self.norm(hidden_states)
            loop_hidden_states.append(hidden_states)
            if output_hidden_states and all_hidden_states:
                all_hidden_states = all_hidden_states[:-1] + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, past_key_values if use_cache else None, all_hidden_states] if v is not None)

        out = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
        )
        out.loop_hidden_states = tuple(loop_hidden_states)
        return out


class Qwen3LoopForCausalLM(Qwen3ForCausalLM):
    """
    Qwen3LoopForCausalLM inherits directly from HF's Qwen3ForCausalLM.
    100% compatible with Qwen3 checkpoints and HF ecosystem.
    """
    config_class = Qwen3LoopConfig

    def __init__(self, config: Qwen3LoopConfig):
        super().__init__(config)
        self.model = Qwen3LoopModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def _supports_default_dynamic_cache(self) -> bool:
        # Com mais de uma passagem o cache tem num_loops x num_hidden_layers
        # camadas virtuais, então o cache padrão do HF não serve.
        return self.config.num_loops == 1 and super()._supports_default_dynamic_cache()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        logits_to_keep: int = 0,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        hidden_states = outputs[0] if not return_dict else outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) and logits_to_keep > 0 else slice(None)

        loss = None
        logits = None

        if labels is not None:
            # Active-token cross-entropy with chunking: compute projection and cross-entropy only on tokens with label != -100.
            # Chunking in blocks of 1024 tokens caps peak logits memory to ~310 MB instead of ~2.5+ GB.
            shift_hidden = hidden_states[:, :-1, :]
            shift_labels = labels[:, 1:]
            active_mask = (shift_labels != -100)

            def _compute_chunked_ce(h_acts, y_acts, chunk_size=1024):
                if not h_acts.size(0):
                    return torch.tensor(0.0, device=h_acts.device, requires_grad=True)
                total_loss = 0.0
                n = h_acts.size(0)
                for start_idx in range(0, n, chunk_size):
                    end_idx = min(start_idx + chunk_size, n)
                    logits_chunk = self.lm_head(h_acts[start_idx:end_idx])
                    loss_chunk = F.cross_entropy(logits_chunk, y_acts[start_idx:end_idx], reduction="sum")
                    total_loss = total_loss + loss_chunk
                return total_loss / max(1, n)

            weights = list(getattr(self.config, "loop_loss_weights", None) or [])
            loop_hidden_states = list(getattr(outputs, "loop_hidden_states", None) or [])
            double_split = bool(getattr(self.config, "enable_double_loop_split", False))
            # Supervisao profunda permanente: sem LoopSplit, as passagens ja incluem
            # a saida final; com LoopSplit, os snapshots sao meio de caminho e a
            # saida final entra com peso 1.0 sempre.
            exits = list(loop_hidden_states)
            if double_split and hidden_states is not None:
                exits = exits + [hidden_states]
            if weights and len(exits) > 1:
                mid_weights = list(weights)
                while len(mid_weights) < len(exits) - 1:
                    mid_weights.append(1.0)
                exit_weights = mid_weights[:len(exits) - 1] + [1.0]

                total_loss = 0.0
                total_weight = 0.0
                norm_layer = getattr(getattr(self, "model", None), "norm", None)
                for w, l_hidden in zip(exit_weights, exits):
                    # Snapshots de segmento saem sem norm final: projecao auxiliar
                    # passa pela mesma RMSNorm (nao toca a recorrencia).
                    if double_split and norm_layer is not None and l_hidden is not hidden_states:
                        l_hidden = norm_layer(l_hidden)
                    if w > 0:
                        shift_l_h = l_hidden[:, :-1, :]
                        if active_mask.any():
                            active_l_h = shift_l_h[active_mask]
                            l_loss = _compute_chunked_ce(active_l_h, shift_labels[active_mask])
                        else:
                            l_loss = torch.tensor(0.0, device=hidden_states.device, requires_grad=True)
                        total_loss = total_loss + (w * l_loss)
                        total_weight += w
                loss = total_loss / max(1e-4, total_weight)
            else:
                if active_mask.any():
                    active_h = shift_hidden[active_mask]
                    loss = _compute_chunked_ce(active_h, shift_labels[active_mask])
                else:
                    loss = torch.tensor(0.0, device=hidden_states.device, requires_grad=True)

            if not self.training:
                logits = self.lm_head(hidden_states[:, slice_indices, :])
        else:
            logits = self.lm_head(hidden_states[:, slice_indices, :])

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        causal_out = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        causal_out.loop_hidden_states = getattr(outputs, "loop_hidden_states", None)
        return causal_out

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        loop_overrides = {
            key: kwargs.pop(key)
            for key in (
                "num_loops",
                "skip_loop_final_norm",
                "enable_double_loop_split",
                "loop_middle_layers",
                "loop_share_kv",
                "loop_loss_weights",
            )
            if key in kwargs
        }

        config = kwargs.pop("config", None)
        if config is None:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        if not isinstance(config, Qwen3LoopConfig):
            config_dict = config.to_dict()
            config_dict.update(loop_overrides)
            config_dict.setdefault("num_loops", 2)
            config_dict["model_type"] = "qwen3loop"
            config = Qwen3LoopConfig.from_dict(config_dict)
        else:
            for key, value in loop_overrides.items():
                setattr(config, key, value)

        kwargs["config"] = config
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)


# Register with Auto classes
try:
    AutoConfig.register("qwen3loop", Qwen3LoopConfig)
    AutoModelForCausalLM.register(Qwen3LoopConfig, Qwen3LoopForCausalLM)
except Exception:
    pass
