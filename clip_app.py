import argparse
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import clip
import torch
from PIL import Image
from torchvision.models import ResNet50_Weights


DEFAULT_MODEL_NAME = "ViT-L/14@336px"


def load_imagenet_labels() -> Iterable[str]:
    """Return the ImageNet-1k label set packaged with torchvision weights metadata."""

    return ResNet50_Weights.IMAGENET1K_V2.meta["categories"]


def load_labels(path: Optional[Path] = None) -> List[str]:
    """Load labels from a file (one per line) or fallback to ImageNet-1k."""

    if path is None:
        return list(load_imagenet_labels())

    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Labels file not found: {resolved}")

    # Strip empty/comment lines; keep as simple list of strings
    labels: List[str] = []
    with resolved.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            labels.append(line)
    if not labels:
        raise ValueError(f"No labels found in file: {resolved}")
    return labels


@lru_cache(maxsize=None)
def load_model(model_name: str, device: str):
    """Load and cache a CLIP model + preprocess pipeline for a given model/device."""

    return clip.load(model_name, device=device)


def rank_labels_from_image(
    image: Image.Image,
    labels: Sequence[str],
    device: str,
    model_name: str = DEFAULT_MODEL_NAME,
    model=None,
    preprocess=None,
) -> List[Tuple[str, float]]:
    """Rank labels against an in-memory image."""

    if model is None or preprocess is None:
        model, preprocess = load_model(model_name, device)

    with torch.inference_mode():
        image_tensor = preprocess(image.convert("RGB")).unsqueeze(0).to(device)
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


def rank_labels(image_path: Path, labels: Iterable[str], device: str) -> List[Tuple[str, float]]:
    resolved_path = image_path.expanduser().resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Image not found: {resolved_path}")

    model, preprocess = load_model(DEFAULT_MODEL_NAME, device)
    return rank_labels_from_image(
        Image.open(resolved_path),
        labels,
        device,
        DEFAULT_MODEL_NAME,
        model,
        preprocess,
    )


def print_results(image_path: Path, ranked_labels: List[Tuple[str, float]], top_k: int) -> None:
    print(f"Label ranking for image: {image_path}")
    for index, (label, score) in enumerate(ranked_labels[:top_k], start=1):
        print(f"{index}. {label:<12} {score:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify an image with OpenAI's CLIP using the ImageNet-1k label set."
    )
    parser.add_argument("image", type=Path, help="Path to the image to evaluate.")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device to use (default: auto-detect CUDA, otherwise CPU).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help="CLIP model to use (default: ViT-L/14@336px).",
    )
    parser.add_argument(
        "--labels-file",
        type=Path,
        default=None,
        help="Optional path to a text file with one label per line. Falls back to ImageNet-1k if omitted.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of predictions to display (default: 5).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = args.device
    labels = load_labels(args.labels_file)
    model, preprocess = load_model(args.model, device)
    ranked = rank_labels_from_image(
        Image.open(args.image.expanduser().resolve()),
        labels,
        device,
        args.model,
        model,
        preprocess,
    )
    print_results(args.image, ranked, args.top_k)


if __name__ == "__main__":
    main()
