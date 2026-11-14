import argparse
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = REPO_ROOT / "models" / "plamo-3-translate-0.8b-hf"
DEFAULT_OUT_PATH = REPO_ROOT / "cpu_reference.json"

TEST_CASES = [
    {
        "name": "en_to_ja_1",
        "source_lang": "en",
        "target_lang": "ja",
        "text": "The meeting has been rescheduled for next Monday.",
    },
    {
        "name": "ja_to_en_1",
        "source_lang": "ja",
        "target_lang": "en",
        "text": "会議は来週の月曜日に変更されました。",
    },
    {
        "name": "en_to_ja_2",
        "source_lang": "en",
        "target_lang": "ja",
        "text": "Artificial intelligence models can now execute efficiently on local NPUs without relying on cloud servers.",
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PyTorch FP32 reference outputs for parity checks")
    parser.add_argument("--model_dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT_PATH)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_dir),
        trust_remote_code=True,
        dtype=torch.float32,
    )
    model.eval()

    results = []
    for case in TEST_CASES:
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": case["text"]}],
            source_lang=case["source_lang"],
            target_lang=case["target_lang"],
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"]
        with torch.no_grad():
            logits = model(input_ids=input_ids).logits[0, -1, :]
            top_vals, top_ids = torch.topk(logits, 10)
            out = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=128,
            )
        gen_ids = out[0, input_ids.shape[1] :].tolist()
        translation = tokenizer.decode(gen_ids, skip_special_tokens=True)
        prompt_str = tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=False)
        print(f"[{case['name']}] prompt len={input_ids.shape[1]}")
        print(f"  Prompt: {prompt_str!r}")
        print(
            f"  Top-5 next tokens: {[(int(i), tokenizer.decode([int(i)]), round(float(v), 4)) for i, v in zip(top_ids[:5], top_vals[:5])]}"
        )
        print(f"  Translation: {translation}")
        results.append(
            {
                "name": case["name"],
                "source_lang": case["source_lang"],
                "target_lang": case["target_lang"],
                "text": case["text"],
                "prompt": prompt_str,
                "input_ids": input_ids[0].tolist(),
                "first_top10_ids": [int(x) for x in top_ids.tolist()],
                "first_top10_vals": [float(x) for x in top_vals.tolist()],
                "generated_ids": gen_ids,
                "translation": translation,
            }
        )

    args.output.write_text(json.dumps({"cases": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved reference outputs to:", args.output)


if __name__ == "__main__":
    main()
