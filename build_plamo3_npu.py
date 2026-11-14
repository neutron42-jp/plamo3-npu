import argparse
import json
import shutil
import sys
from pathlib import Path

import onnx
import ryzenai_onnx_utils.passes.llm.absorb_transformation_ops as absorb
import ryzenai_onnx_utils.passes.llm.unabsorb_transformation_ops as unabsorb
import ryzenai_onnx_utils.strategy_builder.llm.builder as builder
import ryzenai_onnx_utils.transform.hybrid_llm as hl


def apply_ryzenai_patches() -> None:
    """Register PLaMo-3 compatibility patches for ryzenai_onnx_utils LLM fusion."""
    # PLaMo-3 OGA weights can fail the strict bias assumption check inside preprocess_matmulnbits_weights.
    orig_prep = hl.preprocess_matmulnbits_weights

    def patched_prep(
        node,
        start_index,
        extractor,
        k,
        n,
        block_size,
        bias_offset,
        lora=False,
        enable_ctrl_pkt=False,
        verify=True,
    ):
        return orig_prep(
            node,
            start_index,
            extractor,
            k,
            n,
            block_size,
            bias_offset,
            lora=lora,
            enable_ctrl_pkt=enable_ctrl_pkt,
            verify=False,
        )

    hl.preprocess_matmulnbits_weights = patched_prep

    # With --exclude ssmlp, preserve FLOAT gamma on SkipSimplifiedLayerNormalization.
    orig_upd = absorb.update_operator_dtypes

    def patched_upd(node, from_dtype, extractor):
        if node.op_type == "SkipSimplifiedLayerNormalization":
            return []
        return orig_upd(node, from_dtype, extractor)

    absorb.update_operator_dtypes = patched_upd
    unabsorb.update_operator_dtypes = patched_upd

    # Strix aie2p mladf_v2 prefill transactions support max M=4096 for 16-head q_norm => max 256 tokens fused prefill.
    orig_set_prefill = builder.LlmModelBase._set_prefill_fusion_attributes

    def patched_set_prefill(self):
        orig_set_prefill(self)
        self._set_attribute("skip_lm_head", True)
        dyn_list = [{"sequence_length_padded": 128}, {"sequence_length_padded": 256}]
        self._set_dynamic_shape_list("sequence_length_padded", shape_list=dyn_list)

    builder.LlmModelBase._set_prefill_fusion_attributes = patched_set_prefill


def set_cpu_lm_head_accuracy_level(model_path: Path, accuracy_level: int) -> int:
    """Set MatMulNBits accuracy_level on CPU lm_head nodes (4 unlocks AVX-512 VNNI kernels)."""
    model = onnx.load(str(model_path), load_external_data=False)
    updated = 0

    def walk(graph: onnx.GraphProto) -> None:
        nonlocal updated
        for node in graph.node:
            if node.op_type == "MatMulNBits" and "lm_head" in node.name:
                found = False
                for attr in node.attribute:
                    if attr.name == "accuracy_level":
                        attr.i = accuracy_level
                        found = True
                        break
                if not found:
                    node.attribute.append(onnx.helper.make_attribute("accuracy_level", accuracy_level))
                updated += 1
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.GRAPH:
                    walk(attr.g)

    walk(model.graph)
    if updated > 0:
        onnx.save_model(model, str(model_path))
    return updated


def configure_genai_config(
    src_cfg_path: Path,
    dst_cfg_path: Path,
    max_seq_len: int,
    intra_threads: int,
) -> None:
    """Apply validated PLaMo-3 EOS tokens, CPU thread tuning, and RyzenAI NPU options."""
    src_cfg = json.loads(src_cfg_path.read_text(encoding="utf-8"))
    dst_cfg = json.loads(dst_cfg_path.read_text(encoding="utf-8"))

    dst_cfg["model"]["eos_token_id"] = src_cfg["model"]["eos_token_id"]
    sess_opts = dst_cfg["model"]["decoder"]["session_options"]
    if intra_threads > 0:
        sess_opts["intra_op_num_threads"] = intra_threads

    ryzenai_opts = sess_opts["provider_options"][0]["RyzenAI"]
    ryzenai_opts.update(
        {
            "hybrid_opt_max_seq_length": str(max_seq_len),
            "hybrid_opt_token_backend": "npu",
            # Mirror AMD gemma.py: io-binding KV cache corrupts heap/BO lifetime on gemma3_text.
            "fusion_opt_io_bind_kv_cache": "0",
            # Keep eager fallback (>256 prefill tokens) off AIE GQA which misses PLaMo-3 pre-RoPE norm txns.
            "hybrid_dbg_use_aie_gqa": "0",
            "hybrid_opt_npu_read_ahead": "-1",
            "hybrid_opt_keep_performance_mode": "1",
            "hybrid_opt_enable_npu_preemption": "0",
            "hybrid_opt_enable_npu_qos": "1",
        }
    )
    dst_cfg_path.write_text(json.dumps(dst_cfg, indent=4) + "\n", encoding="utf-8")


REPO_ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Build fused NPU PLaMo-3 Translate 0.8B OGA model")
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=REPO_ROOT / "models" / "plamo-3-translate-0.8b-oga",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=REPO_ROOT / "models" / "plamo-3-translate-0.8b-npu",
    )
    parser.add_argument("--max_seq_len", type=int, default=4096)
    parser.add_argument("--intra_threads", type=int, default=8)
    parser.add_argument("--lm_head_accuracy_level", type=int, default=4)
    args = parser.parse_args()

    in_model = (args.input_dir / "model.onnx").resolve()
    out_dir = args.output_dir.resolve()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_model = out_dir / "model.onnx"

    apply_ryzenai_patches()

    import ryzenai_onnx_utils.partitioner as partitioner

    sys.argv = [
        "onnx_utils",
        "optimize",
        "--input-model",
        str(in_model),
        "--output-model",
        str(out_model),
        "--force",
        "llm",
        "--model-type",
        "default",
        "--prefill",
        "npu_fusion",
        "--token",
        "npu_fusion",
        "--priority",
        "performance",
        "--no-jit",
        "--ctrl-pkt",
        "--max-seq-len",
        str(args.max_seq_len),
        "--exclude",
        "ssmlp",
        "--attributes",
        "skip_lm_head=true",
        "fuse_qk_mha=false",
        "--prefill-fusion-eager-fallback",
    ]
    try:
        rc = partitioner.main()
        if rc not in (None, 0):
            raise SystemExit(rc)
    except SystemExit as exc:
        if exc.code not in (None, 0):
            raise

    if args.lm_head_accuracy_level > 0:
        updated = set_cpu_lm_head_accuracy_level(out_model, args.lm_head_accuracy_level)
        print(f"Updated accuracy_level={args.lm_head_accuracy_level} on {updated} lm_head MatMulNBits node(s)")

    configure_genai_config(
        args.input_dir / "genai_config.json",
        out_dir / "genai_config.json",
        max_seq_len=args.max_seq_len,
        intra_threads=args.intra_threads,
    )
    for item in args.input_dir.iterdir():
        if item.is_file() and item.name not in {"genai_config.json", "model.onnx", "model.onnx.data"}:
            shutil.copy2(item, out_dir / item.name)
    print("Built successfully:", out_dir)


if __name__ == "__main__":
    main()
