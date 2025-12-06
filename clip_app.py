import argparse
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import clip
import torch
from PIL import Image


def load_labels(labels: Iterable[str], labels_file: Optional[Path]) -> List[str]:
    collected: List[str] = []

    if labels:
        collected.extend(labels)

    if labels_file:
        text = labels_file.read_text(encoding="utf-8")
        file_labels = [line.strip() for line in text.splitlines() if line.strip()]
        collected.extend(file_labels)

    if not collected:
        raise ValueError("No labels provided. Use --labels or --labels-file to supply labels.")

    return collected


def rank_labels(image_path: Path, labels: List[str], device: str) -> List[Tuple[str, float]]:
    resolved_path = image_path.expanduser().resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Image not found: {resolved_path}")

    model, preprocess = clip.load("ViT-B/32", device=device)

    with torch.inference_mode():
        image_tensor = preprocess(Image.open(resolved_path).convert("RGB")).unsqueeze(0).to(device)
        text_tokens = clip.tokenize(labels).to(device)

        image_features = model.encode_image(image_tensor)
        text_features = model.encode_text(text_tokens)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        logits = 100.0 * image_features @ text_features.T
        probabilities = logits.softmax(dim=-1).squeeze(0)

    scores = probabilities.cpu().tolist()
    ranked = sorted(zip(labels, scores), key=lambda item: item[1], reverse=True)
    return ranked


def print_results(image_path: Path, ranked_labels: List[Tuple[str, float]]) -> None:
    print(f"Label ranking for image: {image_path}")
    for index, (label, score) in enumerate(ranked_labels, start=1):
        print(f"{index}. {label:<12} {score:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score an image against labels using OpenAI's CLIP.")
    parser.add_argument("image", type=Path, help="Path to the image to evaluate.")
    parser.add_argument(
        "--labels",
        nargs="*",
        help="Labels provided inline (space separated). Surround multi-word labels in quotes.",
        default=[],
    )
    parser.add_argument(
        "--labels-file",
        type=Path,
        help="Optional path to a file containing one label per line.",
        default=None,
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device to use (default: auto-detect CUDA, otherwise CPU).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = args.device
    labels = load_labels(args.labels, args.labels_file)
    ranked = rank_labels(args.image, labels, device)
    print_results(args.image, ranked)


if __name__ == "__main__":
    main()
