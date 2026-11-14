import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Download pfnet/plamo-3-translate-0.8b from Hugging Face")
    parser.add_argument("--repo_id", type=str, default="pfnet/plamo-3-translate-0.8b")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=REPO_ROOT / "models" / "plamo-3-translate-0.8b-hf",
    )
    args = parser.parse_args()

    local_dir = args.output_dir.resolve()
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.repo_id,
        local_dir=str(local_dir),
    )
    print("Downloaded to:", local_dir)
    for p in sorted(local_dir.iterdir()):
        print(f"  {p.name}: {p.stat().st_size} bytes")


if __name__ == "__main__":
    main()
