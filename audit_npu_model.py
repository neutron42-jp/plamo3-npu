import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
import onnx


def audit_dir(model_dir: Path):
    print(f"=== Auditing {model_dir} ===")
    for p in sorted(model_dir.iterdir()):
        if p.is_file():
            print(f"  file: {p.name} ({p.stat().st_size:,} bytes)")

    cfg_path = model_dir / "genai_config.json"
    if cfg_path.exists():
        print("\ngenai_config.json:")
        print(cfg_path.read_text(encoding="utf-8"))

    onnx_files = sorted(model_dir.glob("*.onnx"))
    for onnx_path in onnx_files:
        print(f"\n--- ONNX Graph: {onnx_path.name} ---")
        m = onnx.load(str(onnx_path), load_external_data=False)
        counts = Counter()
        by_domain = defaultdict(list)

        def walk(graph: onnx.GraphProto, prefix: str = "root") -> None:
            for node in graph.node:
                dom = node.domain or "ai.onnx"
                key = f"{dom}:{node.op_type}"
                counts[key] += 1
                if dom != "com.ryzenai":
                    by_domain[key].append(f"{prefix}/{node.name}")
                for attr in node.attribute:
                    if attr.type == onnx.AttributeProto.GRAPH:
                        walk(attr.g, f"{prefix}.{node.op_type}.{attr.name}")

        walk(m.graph)

        print("Op counts by domain (including If subgraphs):")
        for k, v in sorted(counts.items()):
            print(f"  {k}: {v}")

        print("\nNon-com.ryzenai node names (sample up to 10 each):")
        for k, names in sorted(by_domain.items()):
            print(f"  {k} ({len(names)}): {names[:10]}")

    cache_dir = model_dir / "cache"
    if cache_dir.is_dir():
        metas = sorted(cache_dir.glob("*_meta.json"))
        if metas:
            print("\n--- DynamicDispatch Cache Summary ---")
            for mp in metas:
                meta = json.loads(mp.read_text(encoding="utf-8"))
                ops = [op["type"] for op in meta.get("op_list", [])]
                print(f"  {mp.name}: {len(ops)} fused NPU ops -> {dict(sorted(Counter(ops).items()))}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_dir",
        type=Path,
        default=Path(__file__).resolve().parent / "models" / "plamo-3-translate-0.8b-npu",
        nargs="?",
    )
    args = parser.parse_args()
    audit_dir(args.model_dir)


if __name__ == "__main__":
    main()
