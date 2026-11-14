import argparse
import ctypes
import importlib.util
import json
import os
import re
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def _find_voe_lib(filename: str, env_var: str | None = None) -> str | None:
    """Locate a Ryzen AI VOE shared library from env, installed voe package, or local .venv."""
    if env_var:
        env_val = os.environ.get(env_var)
        if env_val and Path(env_val).is_file():
            return env_val
    spec = importlib.util.find_spec("voe")
    if spec is not None:
        roots = []
        if spec.origin:
            roots.append(Path(spec.origin).resolve().parent)
        if spec.submodule_search_locations:
            roots.extend(Path(loc).resolve() for loc in spec.submodule_search_locations)
        for root in roots:
            candidate = root / "lib" / filename
            if candidate.is_file():
                return str(candidate)
    venv_root = Path(os.environ.get("RYZEN_AI_VENV", str(REPO_ROOT / ".venv")))
    for candidate in sorted(venv_root.glob(f"lib/python*/site-packages/voe/lib/{filename}")):
        if candidate.is_file():
            return str(candidate)
    return None


DEFAULT_RYZENAI_EP_PATH = _find_voe_lib("libonnxruntime_providers_ryzenai.so", "RYZENAI_EP_PATH")
DEFAULT_RYZEN_MM_PATH = _find_voe_lib("libryzen_mm.so.1")
if DEFAULT_RYZENAI_EP_PATH:
    os.environ.setdefault("RYZENAI_EP_PATH", DEFAULT_RYZENAI_EP_PATH)

import numpy as np
import onnxruntime_genai as og
from transformers import AutoTokenizer

DEFAULT_MODEL_DIR = REPO_ROOT / "models" / "plamo-3-translate-0.8b-npu"
_HF_TOKENIZER_DIR = REPO_ROOT / "models" / "plamo-3-translate-0.8b-hf"
DEFAULT_TOKENIZER_DIR = _HF_TOKENIZER_DIR if _HF_TOKENIZER_DIR.exists() else DEFAULT_MODEL_DIR
DEFAULT_REF_PATH = REPO_ROOT / "cpu_reference.json"
EOS_IDS = {1, 2, 7, 16}
# Ryzen AI 1.8.0 NPU FLATMHA lacks 16-head/dim-128 window=2048 kernels and coerces local_window_size to -1.
# Keeping each NPU forward+decode trajectory strictly below 2048 tokens prevents RoPE local attention aliasing loops.
SAFE_SINGLE_PROMPT_TOKENS = 750
SAFE_TOTAL_SEQ_LEN = 1850


def has_tail_repetition(token_ids: list[int]) -> int:
    """Return repeated tail span length when output locks into an unmistakable repetition loop, else 0."""
    n = len(token_ids)
    for span in range(1, 5):
        reps = 10 if span <= 2 else 6
        need = span * reps
        if n >= need:
            pat = token_ids[-span:]
            if token_ids[-need:] == pat * reps:
                return span * (reps - 1)
    for span in range(5, 49):
        reps = 4
        need = span * reps
        if n >= need:
            pat = token_ids[-span:]
            if token_ids[-need:] == pat * reps:
                return span * (reps - 1)
    return 0


def split_text_into_chunks(
    text: str,
    tokenizer,
    max_chunk_tokens: int = 280,
    first_chunk_tokens: int = 180,
) -> tuple[list[tuple[str, str]], str]:
    """Split text into (leading_separator, chunk_text) pairs plus trailing separator, staying within token budget."""

    def tok_len(s: str) -> int:
        return len(tokenizer.encode(s, add_special_tokens=False))

    def split_oversized(sep: str, body: str) -> list[tuple[str, str]]:
        if tok_len(body) <= max_chunk_tokens:
            return [(sep, body)]
        # Split on single newlines or sentence boundaries while retaining exact whitespace separators.
        sub_parts = re.split(r"(\n+|(?<=[.!?])\s+|(?<=[。！？]))", body)
        out: list[tuple[str, str]] = []
        cur_sep = sep
        cur_buf = ""

        def flush_piece(piece_sep: str, piece: str) -> None:
            if tok_len(piece) <= max_chunk_tokens:
                out.append((piece_sep, piece))
                return
            # Fallback for wall-of-text paragraphs without sentence punctuation: split on spaces/commas, then tokens.
            word_parts = re.split(r"(\s+|(?<=[,、，])\s*)", piece)
            w_sep = piece_sep
            w_buf = ""
            for wp in word_parts:
                if not wp:
                    continue
                if re.fullmatch(r"\s+", wp):
                    if w_buf:
                        out.append((w_sep, w_buf))
                        w_sep = wp
                        w_buf = ""
                    else:
                        w_sep += wp
                elif w_buf and tok_len(w_buf + wp) > max_chunk_tokens:
                    if tok_len(w_buf) <= max_chunk_tokens:
                        out.append((w_sep, w_buf))
                    else:
                        ids = tokenizer.encode(w_buf, add_special_tokens=False)
                        for idx in range(0, len(ids), max_chunk_tokens):
                            sub_s = tokenizer.decode(ids[idx : idx + max_chunk_tokens], skip_special_tokens=True)
                            if sub_s:
                                out.append((w_sep if idx == 0 else "", sub_s))
                    w_sep = ""
                    w_buf = wp
                else:
                    w_buf = w_buf + wp
            if w_buf:
                if tok_len(w_buf) <= max_chunk_tokens:
                    out.append((w_sep, w_buf))
                else:
                    ids = tokenizer.encode(w_buf, add_special_tokens=False)
                    for idx in range(0, len(ids), max_chunk_tokens):
                        sub_s = tokenizer.decode(ids[idx : idx + max_chunk_tokens], skip_special_tokens=True)
                        if sub_s:
                            out.append((w_sep if idx == 0 else "", sub_s))

        for sp in sub_parts:
            if not sp:
                continue
            if re.fullmatch(r"\n+|\s+", sp):
                if cur_buf:
                    flush_piece(cur_sep, cur_buf)
                    cur_sep = sp
                    cur_buf = ""
                else:
                    cur_sep += sp
            elif cur_buf and tok_len(cur_buf + sp) > max_chunk_tokens:
                flush_piece(cur_sep, cur_buf)
                cur_sep = ""
                cur_buf = sp
            else:
                cur_buf = cur_buf + sp
        if cur_buf:
            flush_piece(cur_sep, cur_buf)
        elif out and cur_sep:
            # Should not normally happen unless trailing whitespace inside body
            pass
        return out

    para_parts = re.split(r"(\n[ \t]*\n+)", text)
    raw_units: list[tuple[str, str]] = []
    pending_sep = ""
    for part in para_parts:
        if not part:
            continue
        if re.fullmatch(r"\n[ \t]*\n+", part):
            pending_sep += part
        else:
            raw_units.append((pending_sep, part))
            pending_sep = ""

    fine_units: list[tuple[str, str]] = []
    for sep, body in raw_units:
        fine_units.extend(split_oversized(sep, body))

    packed: list[tuple[str, str]] = []
    cur_sep = ""
    cur_body = ""
    for sep, body in fine_units:
        budget = first_chunk_tokens if not packed else max_chunk_tokens
        if not cur_body:
            cur_sep, cur_body = sep, body
        else:
            candidate = cur_body + sep + body
            if tok_len(candidate) <= budget:
                cur_body = candidate
            else:
                packed.append((cur_sep, cur_body))
                cur_sep, cur_body = sep, body
    if cur_body:
        packed.append((cur_sep, cur_body))
    return packed, pending_sep


def build_chunk_prompt_ids(
    tokenizer,
    body: str,
    history_pair: list[dict[str, str]] | None,
    source_lang: str | None,
    target_lang: str,
    *,
    form: str | None = None,
    context: str | None = None,
    glossary: list[dict[str, str]] | None = None,
    max_prompt_tokens: int = 900,
) -> list[int]:
    """Encode a chunk with up to one recent translation pair, dropping history if prompt exceeds budget."""
    if history_pair:
        ids = encode_prompt(
            tokenizer,
            body,
            source_lang,
            target_lang,
            form=form,
            context=context,
            glossary=glossary,
            messages=history_pair + [{"role": "user", "content": body}],
        )
        if len(ids) <= max_prompt_tokens:
            return ids
    return encode_prompt(
        tokenizer,
        body,
        source_lang,
        target_lang,
        form=form,
        context=context,
        glossary=glossary,
    )


def encode_prompt(
    tokenizer,
    text: str,
    source_lang: str | None,
    target_lang: str,
    *,
    form: str | None = None,
    context: str | None = None,
    glossary: list[dict[str, str]] | None = None,
    messages: list[dict[str, str]] | None = None,
) -> list[int]:
    """Apply the PLaMo-3 translation chat template and return token IDs."""
    kwargs: dict = {
        "source_lang": source_lang,
        "target_lang": target_lang,
        "add_generation_prompt": True,
        "return_dict": True,
        "return_tensors": "np",
    }
    if form and target_lang == "ja":
        kwargs["form"] = form
    if context:
        kwargs["context"] = context
    if glossary:
        kwargs["glossary"] = glossary
    chat_messages = messages if messages is not None else [{"role": "user", "content": text}]
    encoded = tokenizer.apply_chat_template(chat_messages, **kwargs)
    return encoded["input_ids"][0].astype(np.int64).tolist()


def decode_generated_tokens(tokenizer, seq: list[int]) -> tuple[list[int], str]:
    """Stop at the first PLaMo-3 EOS token ({1, 2, 7, 16}) and decode to text."""
    trimmed = []
    for tok in seq:
        tok_id = int(tok)
        if tok_id in EOS_IDS:
            break
        trimmed.append(tok_id)
    return trimmed, tokenizer.decode(trimmed, skip_special_tokens=True)


def reset_token_kv_state(model: og.Model) -> int:
    """Advance Token DD attention_mask length away from a collision point via a 2-token step."""
    params = og.GeneratorParams(model)
    params.set_search_options(do_sample=False, max_length=5)
    gen = og.Generator(model, params)
    gen.append_tokens([1, 1])
    gen.generate_next_token()
    gen.generate_next_token()
    del gen, params
    return 3


def run_generation(
    model: og.Model,
    tokenizer,
    input_ids: list[int],
    max_new_tokens: int,
    inspect_first_logits: bool = False,
    last_token_attn_len: int | None = None,
) -> dict:
    """Execute prefill + greedy autoregressive decoding on the loaded NPU model."""
    # Prevent DynamicDispatch Token KV cache sync skip when new_prompt_len + 1 == prev_token_attn_len + 1.
    if last_token_attn_len is not None and abs(len(input_ids) - last_token_attn_len) <= 1:
        reset_token_kv_state(model)

    params = og.GeneratorParams(model)
    params.set_search_options(do_sample=False, max_length=len(input_ids) + max_new_tokens)

    gen = og.Generator(model, params)

    t_prefill_0 = time.perf_counter()
    gen.append_tokens(input_ids)
    t_prefill_1 = time.perf_counter()

    top5_ids: list[int] = []
    top5_vals: list[float] = []
    if inspect_first_logits:
        logits = np.array(gen.get_logits(), copy=True)[0, -1, :]
        top5_ids = np.argsort(-logits)[:5].tolist()
        top5_vals = [float(logits[i]) for i in top5_ids]

    step_ms: list[float] = []
    gen_tokens: list[int] = []
    rep_trim = 0
    t_dec_0 = time.perf_counter()
    while not gen.is_done():
        ts = time.perf_counter()
        gen.generate_next_token()
        te = time.perf_counter()
        step_ms.append((te - ts) * 1000.0)
        next_token = int(gen.get_next_tokens()[0])
        if next_token in EOS_IDS:
            break
        gen_tokens.append(next_token)
        if not inspect_first_logits and len(gen_tokens) >= 16:
            rep_trim = has_tail_repetition(gen_tokens)
            if rep_trim:
                break
    t_dec_1 = time.perf_counter()

    full_seq = gen.get_sequence(0)
    raw_gen = [int(t) for t in full_seq[len(input_ids) :]]
    trimmed_ids, text = decode_generated_tokens(tokenizer, raw_gen)
    if rep_trim and len(trimmed_ids) > rep_trim:
        trimmed_ids = trimmed_ids[:-rep_trim]
        text = tokenizer.decode(trimmed_ids, skip_special_tokens=True)
    del gen, params

    prefill_ms = (t_prefill_1 - t_prefill_0) * 1000.0
    decode_s = max(t_dec_1 - t_dec_0, 1e-9)
    decode_ms = decode_s * 1000.0
    num_tokens = len(raw_gen)
    tok_per_s = num_tokens / decode_s
    next_last_attn = len(input_ids) + len(step_ms) - 1 if len(step_ms) >= 2 else last_token_attn_len

    return {
        "prompt_tokens": len(input_ids),
        "generated_tokens": num_tokens,
        "output_tokens": len(trimmed_ids),
        "translation": text,
        "prefill_ms": prefill_ms,
        "decode_ms": decode_ms,
        "tok_per_s": tok_per_s,
        "step_ms": step_ms,
        "last_token_attn_len": next_last_attn,
        "top5_ids": top5_ids,
        "top5_vals": top5_vals,
    }


def run_xrt_smi_examine() -> None:
    """Capture active Ryzen AI NPU device and hardware context status via xrt-smi."""
    xrt_smi = Path("/opt/xilinx/xrt/bin/xrt-smi")
    if not xrt_smi.exists():
        print("\n[xrt-smi] Not found at /opt/xilinx/xrt/bin/xrt-smi")
        return
    print("\n=== Active NPU Hardware Telemetry (xrt-smi examine) ===")
    proc = subprocess.run([str(xrt_smi), "examine"], capture_output=True, text=True)
    if proc.stdout:
        print(proc.stdout.rstrip())
    if proc.stderr:
        print(proc.stderr.rstrip())


def configure_ryzenai_runtime() -> None:
    """Disable direct DRM BO allocations so RyzenMM uses pin_user_pages-compatible memory."""
    ep_path = DEFAULT_RYZENAI_EP_PATH or _find_voe_lib("libonnxruntime_providers_ryzenai.so", "RYZENAI_EP_PATH")
    mm_path = DEFAULT_RYZEN_MM_PATH or _find_voe_lib("libryzen_mm.so.1")
    if not ep_path or not mm_path:
        raise FileNotFoundError(
            "Ryzen AI VOE libraries not found. Source env.sh or set RYZEN_AI_VENV / RYZENAI_EP_PATH."
        )
    os.environ.setdefault("RYZENAI_EP_PATH", ep_path)
    rmm = ctypes.CDLL(mm_path)
    rmm.ryzen_mm_platform_xrt_set_direct_bo_allocations.argtypes = [ctypes.c_bool]
    rmm.ryzen_mm_platform_xrt_set_direct_bo_allocations.restype = ctypes.c_int
    rc = rmm.ryzen_mm_platform_xrt_set_direct_bo_allocations(False)
    if rc != 0:
        raise RuntimeError(f"ryzen_mm_platform_xrt_set_direct_bo_allocations(False) failed with code {rc}")


BENCHMARK_TEXT = (
    "The AMD Ryzen AI neural processing unit accelerates transformer inference by fusing layer normalization, "
    "query-key-value projections, rotary position embeddings, multi-head attention, and gated MLP blocks into "
    "compact control-packet dispatches on XDNA2 hardware while maintaining numerical equivalence with CPU execution. "
    "When optimizing latency and throughput, developers must profile prefill versus token decode, eliminate redundant "
    "host-device synchronization, align tensor dimensions to hardware block sizes, and verify accuracy on reference prompts."
)


def configure_ryzenai_options(cfg: og.Config, model_dir: Path) -> None:
    """Apply validated RyzenAI NPU provider options unless using a non-NPU config."""
    cfg_json = json.loads((model_dir / "genai_config.json").read_text(encoding="utf-8"))
    providers = cfg_json.get("model", {}).get("decoder", {}).get("session_options", {}).get("provider_options", [])
    has_ryzenai = any("RyzenAI" in p for p in providers if isinstance(p, dict))
    if not has_ryzenai:
        return
    cfg.set_provider_option("RyzenAI", "hybrid_dbg_use_aie_gqa", "0")
    cfg.set_provider_option("RyzenAI", "fusion_opt_io_bind_kv_cache", "0")
    cfg.set_provider_option("RyzenAI", "hybrid_opt_npu_read_ahead", "-1")
    cfg.set_provider_option("RyzenAI", "hybrid_opt_keep_performance_mode", "1")
    cfg.set_provider_option("RyzenAI", "hybrid_opt_enable_npu_preemption", "0")
    cfg.set_provider_option("RyzenAI", "hybrid_opt_enable_npu_qos", "1")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PLaMo-3 Translate 0.8B purely on AMD Ryzen AI NPU")
    parser.add_argument("--model_dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--tokenizer_dir", type=Path, default=DEFAULT_TOKENIZER_DIR)
    parser.add_argument("--source_lang", type=str, default="en")
    parser.add_argument("--target_lang", type=str, default="ja")
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--verify_ref", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--ref_path", type=Path, default=DEFAULT_REF_PATH)
    parser.add_argument("--max_new_tokens", type=int, default=160)
    args = parser.parse_args()

    configure_ryzenai_runtime()
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer_dir), trust_remote_code=True)
    cfg = og.Config(str(args.model_dir))
    configure_ryzenai_options(cfg, args.model_dir)
    model = og.Model(cfg)

    try:
        if args.verify_ref:
            ref = json.loads(args.ref_path.read_text(encoding="utf-8"))
            all_top1_match = True
            last_attn = None
            for case in ref["cases"]:
                input_ids = [int(t) for t in case["input_ids"]]
                res = run_generation(
                    model,
                    tokenizer,
                    input_ids,
                    max_new_tokens=args.max_new_tokens,
                    inspect_first_logits=True,
                    last_token_attn_len=last_attn,
                )
                last_attn = res["last_token_attn_len"]
                ref_top1 = int(case["first_top10_ids"][0])
                npu_top1 = res["top5_ids"][0]
                match_top1 = ref_top1 == npu_top1
                if not match_top1 or not res["translation"].strip() or res["output_tokens"] < 3:
                    all_top1_match = False

                print(f"\n[{case['name']}] ({case['source_lang']} -> {case['target_lang']})")
                print(f"  Input      : {case['text']}")
                print(
                    f"  Ref top-1  : ({ref_top1}, {tokenizer.decode([ref_top1])!r}, {case['first_top10_vals'][0]:.4f})"
                )
                print(
                    f"  NPU top-1  : ({npu_top1}, {tokenizer.decode([npu_top1])!r}, {res['top5_vals'][0]:.4f}) match={match_top1}"
                )
                print(f"  Ref output : {case['translation']}")
                print(f"  NPU output : {res['translation']}")
                print(
                    f"  Timing     : prompt={res['prompt_tokens']} tok, gen={res['generated_tokens']} tok, "
                    f"prefill={res['prefill_ms']:.2f} ms, decode={res['decode_ms']:.2f} ms ({res['tok_per_s']:.2f} tok/s)"
                )

            run_xrt_smi_examine()

            if not all_top1_match:
                raise SystemExit("NPU verification FAILED: top-1 next token mismatch or incomplete translation.")
            print("\nNPU verification PASSED (all 3 cases match reference top-1 logits and produced valid translations).")
            if not args.benchmark:
                return

        if args.benchmark:
            bench_ids = encode_prompt(tokenizer, BENCHMARK_TEXT, "en", "ja")
            prefs: list[float] = []
            steady_steps: list[float] = []
            last_attn = None
            bench_out = ""
            for run_idx in range(4):
                res = run_generation(
                    model,
                    tokenizer,
                    bench_ids,
                    max_new_tokens=args.max_new_tokens,
                    inspect_first_logits=False,
                    last_token_attn_len=last_attn,
                )
                last_attn = res["last_token_attn_len"]
                if run_idx > 0:
                    prefs.append(res["prefill_ms"])
                    # Skip step[0] (0.06 ms prefill sample) and step[1] if cold warmup, though r>0 is warm.
                    steady_steps.extend(res["step_ms"][2:])
                    bench_out = res["translation"]
            arr = np.array(steady_steps)
            print(f"\n=== Long-Output NPU Benchmark (prompt={len(bench_ids)} tok, 3 warm runs) ===")
            print(f"  Prefill (mean) : {np.mean(prefs):.2f} ms")
            print(
                f"  Decode steady  : mean={arr.mean():.2f} ms ({1000.0 / arr.mean():.2f} tok/s), "
                f"median={np.median(arr):.2f} ms ({1000.0 / np.median(arr):.2f} tok/s), min={arr.min():.2f} ms"
            )
            print(f"  Translation    : {bench_out}")
            return

        text = args.text
        if not text:
            text = "The meeting has been rescheduled for next Monday."
        input_ids = encode_prompt(tokenizer, text, args.source_lang, args.target_lang)
        if len(input_ids) <= SAFE_SINGLE_PROMPT_TOKENS:
            allowed_new = min(max(args.max_new_tokens, 1), max(1, SAFE_TOTAL_SEQ_LEN - len(input_ids)))
            res = run_generation(
                model,
                tokenizer,
                input_ids,
                max_new_tokens=allowed_new,
                inspect_first_logits=False,
            )
            print(f"Source ({args.source_lang}): {text}")
            print(f"Translation ({args.target_lang}): {res['translation']}")
            print(
                f"[NPU Timing] prompt={res['prompt_tokens']} tok, gen={res['generated_tokens']} tok, "
                f"prefill={res['prefill_ms']:.2f} ms, decode={res['decode_ms']:.2f} ms ({res['tok_per_s']:.2f} tok/s)"
            )
        else:
            chunks, trailing_sep = split_text_into_chunks(text, tokenizer)
            out_parts: list[str] = []
            history_pair: list[dict[str, str]] | None = None
            last_attn = None
            total_prompt = 0
            total_gen = 0
            total_prefill_ms = 0.0
            total_decode_ms = 0.0
            chunk_req_budget = args.max_new_tokens if args.max_new_tokens != 160 else 512
            for sep, body in chunks:
                chunk_ids = build_chunk_prompt_ids(
                    tokenizer, body, history_pair, args.source_lang, args.target_lang
                )
                allowed_new = min(max(chunk_req_budget, 16), max(16, SAFE_TOTAL_SEQ_LEN - len(chunk_ids)))
                res = run_generation(
                    model,
                    tokenizer,
                    chunk_ids,
                    max_new_tokens=allowed_new,
                    inspect_first_logits=False,
                    last_token_attn_len=last_attn,
                )
                last_attn = res["last_token_attn_len"]
                trans = res["translation"]
                out_parts.append(sep + trans)
                history_pair = [{"role": "user", "content": body}, {"role": "assistant", "content": trans}]
                total_prompt += res["prompt_tokens"]
                total_gen += res["generated_tokens"]
                total_prefill_ms += res["prefill_ms"]
                total_decode_ms += res["decode_ms"]
            full_translation = "".join(out_parts) + trailing_sep
            tok_s = total_gen / max(total_decode_ms / 1000.0, 1e-9)
            print(f"Source ({args.source_lang}): {text}")
            print(f"Translation ({args.target_lang}): {full_translation}")
            print(
                f"[NPU Timing] chunks={len(chunks)}, prompt={total_prompt} tok, gen={total_gen} tok, "
                f"prefill={total_prefill_ms:.2f} ms, decode={total_decode_ms:.2f} ms ({tok_s:.2f} tok/s)"
            )
    finally:
        del model


if __name__ == "__main__":
    main()
