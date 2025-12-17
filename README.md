# CLIP Image Recognition

This repository provides both a command-line tool **and** a simple web front end that use OpenAI's CLIP model to classify images **without** supplying your own labels. Both flows use the ImageNet-1k category set bundled with torchvision and return the most likely matches for a given image.

## Features
- Loads the pretrained `ViT-L/14@336px` CLIP model by default (switchable with `--model`).
- Uses the 1,000-class ImageNet label set out of the box—or point to your own labels file.
- Normalizes embeddings and returns a ranked list of label probabilities.
- Web upload form that classifies an image and shows the top predictions.

## Requirements
- Python 3.9+
- `torch` (with CUDA support if available)
- `clip` (OpenAI CLIP implementation)
- `Pillow`
- `Flask`

Install dependencies (uses the published `openai-clip` wheel rather than cloning from GitHub):

```bash
pip install -r requirements.txt
```

If you see network errors when installing directly from GitHub, ensure you are on a network that permits outbound HTTPS traffic.

> **Note:** The `openai-clip` wheel is pulled from PyPI and may download model weights on first run.

## CLI usage

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

Use a different CLIP model (e.g., smaller/faster) and a custom labels file (one label per line):

```bash
python clip_app.py path/to/image.jpg --model ViT-B/32 --labels-file labels.txt
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

## Web app usage

Start the Flask server:

```bash
export FLASK_APP=web_app.py
flask run
```

Then open http://127.0.0.1:5000 in your browser. Upload any image and the page will display the top 5 ImageNet predictions with their probabilities.

To customize the model or labels for the web app, set env vars before running:

```bash
export CLIP_MODEL_NAME="ViT-L/14@336px"
export CLIP_LABELS_FILE="/full/path/to/labels.txt"  # optional, one label per line
flask run
```

## Development
- Keep imports at the top of the file; no try/except around imports.
- The CLI is self-contained; extend it with additional helpers as needed.

## License
MIT
