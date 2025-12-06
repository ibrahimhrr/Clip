# CLIP Image Recognition CLI

This repository provides a lightweight command-line tool that uses OpenAI's CLIP model to score images against textual labels. Provide an image and a set of candidate labels and receive similarity scores that indicate which label best matches the image.

## Features
- Loads the pretrained `ViT-B/32` CLIP model.
- Accepts labels via command line or a labels file (one label per line).
- Normalizes embeddings and returns a ranked list of label probabilities.

## Requirements
- Python 3.9+
- `torch` (with CUDA support if available)
- `clip` (OpenAI CLIP implementation)
- `Pillow`

Install dependencies:

```bash
pip install -r requirements.txt
```

> **Note:** The `clip` package is pulled directly from the OpenAI GitHub repository.

## Usage

Run the CLI by passing an image path and one or more labels:

```bash
python clip_app.py path/to/image.jpg --labels "a dog" "a cat" "a car"
```

To supply labels from a file (one label per line):

```bash
python clip_app.py path/to/image.jpg --labels-file labels.txt
```

Example output:

```
Label ranking for image: path/to/image.jpg
1. a cat        0.71
2. a dog        0.20
3. a car        0.09
```

Use `--device` to force CPU or CUDA:

```bash
python clip_app.py path/to/image.jpg --labels "a horse" "a person" --device cuda
```

## Development
- Keep imports at the top of the file; no try/except around imports.
- The CLI is self-contained; extend it with additional helpers as needed.

## License
MIT
