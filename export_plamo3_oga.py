import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import onnx_ir as ir
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM

from onnxruntime_genai.models.builders.gemma import Gemma3Model
from onnxruntime_genai.models.builders.base import to_torch_dtype

ORIG_HIDDEN = 1536
PADDED_HIDDEN = 2048
PAD_HIDDEN = PADDED_HIDDEN - ORIG_HIDDEN
RMS_RATIO = ORIG_HIDDEN / PADDED_HIDDEN
RMS_SCALE = math.sqrt(RMS_RATIO)


class Plamo3Model(Gemma3Model):
    """Custom OGA builder mapping PLaMo-3 Translate 0.8B onto Gemma-3 style decoder blocks."""

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        self.hf_config = config
        self.hf_dir = extra_options["hf_dir"]
        self.window_NotLocal_pattern = getattr(config, "sliding_window_pattern", 8)
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)
        self.model_type = "gemma3_text"
        # config.scale_embedding is False for plamo-3-translate-0.8b
        self.embed_attrs["scale"] = 1
        # Per-module RMSNorm offsets (1.0, 1/5, 1/5^1.5) are folded into weights in load_weights()
        self.layernorm_attrs["add_offset"] = 0
        self.attention_attrs["scale"] = 1.0 / float(np.sqrt(self.head_size))

    def is_local(self, layer_id):
        return bool((layer_id + 1) % self.window_NotLocal_pattern)

    def make_rotary_embedding_multi_cache(self):
        hf_dir_str = str(self.hf_dir)
        if hf_dir_str not in sys.path:
            sys.path.insert(0, hf_dir_str)
        from modeling_plamo import RotaryEmbedding

        dummy = torch.zeros(1, dtype=to_torch_dtype(self.io_dtype))

        self.cos_cache_global_name = "cos_cache_global"
        self.sin_cache_global_name = "sin_cache_global"
        rope_global = RotaryEmbedding(
            self.head_size,
            max_position_embeddings=self.context_length,
            parameter=self.hf_config.rope_parameters["full_attention"],
        )
        cos_g, sin_g = rope_global(dummy, seq_len=self.context_length)
        cos_g = cos_g.squeeze().to(to_torch_dtype(self.io_dtype))[:, : self.head_size // 2].contiguous()
        sin_g = sin_g.squeeze().to(to_torch_dtype(self.io_dtype))[:, : self.head_size // 2].contiguous()
        self.make_initializer(cos_g, self.cos_cache_global_name)
        self.make_initializer(sin_g, self.sin_cache_global_name)

        self.cos_cache_local_name = "cos_cache_local"
        self.sin_cache_local_name = "sin_cache_local"
        rope_local = RotaryEmbedding(
            self.head_size,
            max_position_embeddings=self.context_length,
            parameter=self.hf_config.rope_parameters["sliding_attention"],
        )
        cos_l, sin_l = rope_local(dummy, seq_len=self.context_length)
        cos_l = cos_l.squeeze().to(to_torch_dtype(self.io_dtype))[:, : self.head_size // 2].contiguous()
        sin_l = sin_l.squeeze().to(to_torch_dtype(self.io_dtype))[:, : self.head_size // 2].contiguous()
        self.make_initializer(cos_l, self.cos_cache_local_name)
        self.make_initializer(sin_l, self.sin_cache_local_name)

        self.rope_attrs["create_caches"] = False

    def load_weights(self, input_path):
        hf_model = AutoModelForCausalLM.from_pretrained(
            input_path,
            trust_remote_code=True,
            dtype=torch.float32,
        )
        hf_model.eval()

        # Fold per-module RMSNorm offsets directly into weights so ONNX SimplifiedLayerNorm
        # executes with add_offset=0, and apply invariant 1536 -> 2048 hidden_size padding.
        for mod in hf_model.modules():
            if hasattr(mod, "offset") and isinstance(getattr(mod, "weight", None), nn.Parameter):
                with torch.no_grad():
                    mod.weight.data.add_(float(mod.offset))
                mod.offset = 0.0
                # Scale & pad only 1536-dim residual norms; leave 128-dim q_norm/k_norm unchanged.
                if mod.weight.shape == (ORIG_HIDDEN,):
                    mod.weight.data.mul_(RMS_SCALE)
                    mod.weight = nn.Parameter(
                        F.pad(mod.weight.data, (0, PAD_HIDDEN)),
                        requires_grad=False,
                    )

        hf_model.model.embed_tokens.weight = nn.Parameter(
            F.pad(hf_model.model.embed_tokens.weight.data, (0, PAD_HIDDEN)),
            requires_grad=False,
        )
        hf_model.model.embed_tokens.embedding_dim = PADDED_HIDDEN

        # Alias Plamo3Decoder.layers and per-layer submodules to Gemma3Model conventions.
        hf_model.model.layers = hf_model.model.layers.layers
        for layer in hf_model.model.layers:
            mixer = layer.mixer
            q_dim = mixer.q_proj_dim
            k_dim = mixer.k_proj_dim
            v_dim = mixer.v_proj_dim

            w_qkv = mixer.qkv_proj.weight.data
            w_q, w_k, w_v = torch.split(w_qkv, [q_dim, k_dim, v_dim], dim=0)

            mixer.q_proj = nn.Linear(PADDED_HIDDEN, q_dim, bias=False)
            mixer.q_proj.weight = nn.Parameter(F.pad(w_q, (0, PAD_HIDDEN)), requires_grad=False)
            # Replicate the single KV head (1 -> kv_heads) to match Ryzen AI NPU kernel shapes.
            kv_heads = int(self.hf_config.num_key_value_heads)
            w_k_pad = F.pad(w_k, (0, PAD_HIDDEN))
            w_v_pad = F.pad(w_v, (0, PAD_HIDDEN))
            mixer.k_proj = nn.Linear(PADDED_HIDDEN, k_dim * kv_heads, bias=False)
            mixer.k_proj.weight = nn.Parameter(
                torch.cat([w_k_pad] * kv_heads, dim=0).contiguous(),
                requires_grad=False,
            )
            mixer.v_proj = nn.Linear(PADDED_HIDDEN, v_dim * kv_heads, bias=False)
            mixer.v_proj.weight = nn.Parameter(
                torch.cat([w_v_pad] * kv_heads, dim=0).contiguous(),
                requires_grad=False,
            )
            del mixer.qkv_proj

            mixer.o_proj.weight = nn.Parameter(
                F.pad(mixer.o_proj.weight.data, (0, 0, 0, PAD_HIDDEN)),
                requires_grad=False,
            )
            mixer.o_proj.out_features = PADDED_HIDDEN

            mlp = layer.mlp
            w_gate, w_up = mlp.gate_up_proj.weight.data.chunk(2, dim=0)
            mlp.gate_proj = nn.Linear(PADDED_HIDDEN, mlp.intermediate_size, bias=False)
            mlp.gate_proj.weight = nn.Parameter(F.pad(w_gate, (0, PAD_HIDDEN)), requires_grad=False)
            mlp.up_proj = nn.Linear(PADDED_HIDDEN, mlp.intermediate_size, bias=False)
            mlp.up_proj.weight = nn.Parameter(F.pad(w_up, (0, PAD_HIDDEN)), requires_grad=False)
            del mlp.gate_up_proj

            mlp.down_proj.weight = nn.Parameter(
                F.pad(mlp.down_proj.weight.data, (0, 0, 0, PAD_HIDDEN)),
                requires_grad=False,
            )
            mlp.down_proj.out_features = PADDED_HIDDEN

            layer.self_attn = mixer
            layer.input_layernorm = layer.pre_mixer_norm
            layer.post_attention_layernorm = layer.post_mixer_norm
            layer.pre_feedforward_layernorm = layer.pre_mlp_norm
            layer.post_feedforward_layernorm = layer.post_mlp_norm

        hf_model.lm_head.weight = nn.Parameter(
            F.pad(hf_model.lm_head.weight.data, (0, PAD_HIDDEN)),
            requires_grad=False,
        )
        hf_model.lm_head.in_features = PADDED_HIDDEN

        return hf_model


REPO_ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="Export pfnet/plamo-3-translate-0.8b to OGA INT4 ONNX")
    parser.add_argument(
        "--hf_dir",
        type=Path,
        default=REPO_ROOT / "models" / "plamo-3-translate-0.8b-hf",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=REPO_ROOT / "models" / "plamo-3-translate-0.8b-oga",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        default=REPO_ROOT / "models" / ".oga_cache",
    )
    parser.add_argument("--context_length", type=int, default=4096)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--kv_heads", type=int, default=8)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    config = AutoConfig.from_pretrained(str(args.hf_dir), trust_remote_code=True)
    config.hidden_size = PADDED_HIDDEN
    config.rms_norm_eps = float(config.rms_norm_eps) * RMS_RATIO
    config.max_position_embeddings = args.context_length
    config.head_dim = 128
    config.num_key_value_heads = args.kv_heads
    config.query_pre_attn_scalar = 128
    config.hidden_activation = "silu"
    config.final_logit_softcapping = None
    config.attn_logit_softcapping = None

    extra_options = {
        "int4_block_size": str(args.block_size),
        "int4_is_symmetric": True,
        "int4_algo_config": "default",
        "hf_dir": str(args.hf_dir),
        "hf_remote": True,
        "hf_token": "false",
    }

    onnx_model = Plamo3Model(
        config=config,
        io_dtype=ir.DataType.FLOAT16,
        onnx_dtype=ir.DataType.INT4,
        ep="dml",
        cache_dir=str(args.cache_dir),
        extra_options=extra_options,
    )
    onnx_model.make_model(str(args.hf_dir))
    onnx_model.save_model(str(args.out_dir))
    onnx_model.make_genai_config(str(args.hf_dir), {}, str(args.out_dir))
    onnx_model.save_processing(str(args.hf_dir), {}, str(args.out_dir))

    # Ensure genai_config.json uses PLaMo-3 multi-EOS ([1, 16]) and exact context_length
    cfg_path = args.out_dir / "genai_config.json"
    genai_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    genai_cfg["model"]["type"] = "gemma3_text"
    genai_cfg["model"]["context_length"] = args.context_length
    genai_cfg["model"]["eos_token_id"] = [1, 16]
    genai_cfg["search"]["max_length"] = args.context_length
    cfg_path.write_text(json.dumps(genai_cfg, indent=4) + "\n", encoding="utf-8")

    print("Export complete:", args.out_dir)
    for p in sorted(args.out_dir.iterdir()):
        print(f"  {p.name}: {p.stat().st_size} bytes")


if __name__ == "__main__":
    main()
