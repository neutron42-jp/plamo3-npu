import argparse
import json
from pathlib import Path
import numpy as np
import onnx
import onnxruntime_genai as og
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_OGA_DIR = REPO_ROOT / "models" / "plamo-3-translate-0.8b-oga"
DEFAULT_REF_PATH = REPO_ROOT / "cpu_reference.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify exported OGA INT4 ONNX model on CPU against reference outputs")
    parser.add_argument("--oga_dir", type=Path, default=DEFAULT_OGA_DIR)
    parser.add_argument("--ref_path", type=Path, default=DEFAULT_REF_PATH)
    args = parser.parse_args()

    # 1. Audit ONNX ops in exported OGA model
    m = onnx.load(str(args.oga_dir / "model.onnx"), load_external_data=False)
    op_counts = {}
    for node in m.graph.node:
        key = f"{node.domain or 'ai.onnx'}:{node.op_type}"
        op_counts[key] = op_counts.get(key, 0) + 1
    print("Exported OGA op counts:")
    for k, v in sorted(op_counts.items()):
        print(f"  {k}: {v}")

    # 2. Verify OGA INT4 model outputs on CPU against PyTorch float32 reference
    ref = json.loads(args.ref_path.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(str(args.oga_dir), trust_remote_code=True)

    cfg = og.Config(str(args.oga_dir))
    cfg.clear_providers()  # Force CPUExecutionProvider baseline parity check
    model = og.Model(cfg)

    all_ok = True
    for case in ref["cases"]:
        input_ids = case["input_ids"]
        params = og.GeneratorParams(model)
        params.set_search_options(do_sample=False, max_length=len(input_ids) + 128)
        gen = og.Generator(model, params)
        gen.append_tokens(input_ids)

        logits = np.array(gen.get_logits())[0, -1, :]
        top5_ids = np.argsort(-logits)[:5].tolist()
        top5_vals = [float(logits[i]) for i in top5_ids]

        gen.generate_next_token()
        while not gen.is_done():
            gen.generate_next_token()

        seq = gen.get_sequence(0)[len(input_ids) :]
        eos_set = {1, 2, 7, 16}
        trimmed = [int(t) for t in seq if int(t) not in eos_set]
        text = tokenizer.decode(trimmed, skip_special_tokens=True)

        ref_top1 = case["first_top10_ids"][0]
        oga_top1 = top5_ids[0]
        match_top1 = ref_top1 == oga_top1
        print(f"\n[{case['name']}]")
        print(f"  Ref top-1: ({ref_top1}, {tokenizer.decode([ref_top1])!r}, {case['first_top10_vals'][0]:.4f})")
        print(f"  OGA top-1: ({oga_top1}, {tokenizer.decode([oga_top1])!r}, {top5_vals[0]:.4f}) match={match_top1}")
        print(f"  Ref translation: {case['translation']}")
        print(f"  OGA translation: {text}")
        if not match_top1:
            all_ok = False

    if not all_ok:
        raise SystemExit("OGA top-1 mismatch against CPU reference!")
    print("\nOGA CPU verification PASSED (all top-1 next tokens match reference).")


if __name__ == "__main__":
    main()
