# CLIP Image Recognition CLI

This repository provides a lightweight command-line tool that uses OpenAI's CLIP model to classify images **without** supplying your own labels. The CLI uses the ImageNet-1k category set bundled with torchvision and returns the most likely matches for a given image.

## Features
- Loads the pretrained `ViT-B/32` CLIP model.
- Uses the 1,000-class ImageNet label set out of the box—no manual labels required.
- Normalizes embeddings and returns a ranked list of label probabilities.

## Requirements
- Python 3.9+
- `torch` (with CUDA support if available)
- `clip` (OpenAI CLIP implementation)
- `Pillow`

Install dependencies (uses the published `openai-clip` wheel rather than cloning from GitHub):

```bash
pip install -r requirements.txt
```

If you see network errors when installing directly from GitHub, ensure you are on a network that permits outbound HTTPS traffic.

> **Note:** The `clip` package is pulled directly from the OpenAI GitHub repository.

## Usage

Run the CLI by passing an image path; the tool will score it against the ImageNet-1k labels automatically:

```bash
python clip_app.py path/to/image.jpg
```

Quick-start with a sample image (downloads a cat photo and runs the classifier):

```bash
curl -L -o sample.jpg https://images.unsplash.com/photo-1518791841217-8f162f1e1131?w=512
python clip_app.py sample.jpg
```

Show the top 10 predictions:

```bash
python clip_app.py path/to/image.jpg --top-k 10
```

Example output:

```
Label ranking for image: path/to/image.jpg
1. tabby, tabby cat  0.42
2. tiger cat         0.25
3. Egyptian cat      0.18
4. lynx              0.05
5. cougar            0.03
```

Use `--device` to force CPU or CUDA:

```bash
python clip_app.py path/to/image.jpg --device cuda
```

## Development
- Keep imports at the top of the file; no try/except around imports.
- The CLI is self-contained; extend it with additional helpers as needed.

## License
MIT
