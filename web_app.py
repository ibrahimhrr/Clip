from io import BytesIO
from typing import Optional

from flask import Flask, render_template_string, request
from PIL import Image
import torch

from clip_app import load_imagenet_labels, load_model, rank_labels_from_image

app = Flask(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL, PREPROCESS = load_model(DEVICE)
LABELS = list(load_imagenet_labels())
DEFAULT_TOP_K = 5


PAGE_TEMPLATE = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>CLIP Image Recognizer</title>
    <style>
      body { font-family: Arial, sans-serif; margin: 2rem; }
      .container { max-width: 720px; margin: auto; }
      form { border: 1px solid #ddd; padding: 1.5rem; border-radius: 8px; }
      .predictions { margin-top: 1.5rem; }
      table { border-collapse: collapse; width: 100%; }
      th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
      th { background-color: #f7f7f7; }
      .error { color: #b00020; margin-top: 0.5rem; }
    </style>
  </head>
  <body>
    <div class="container">
      <h1>CLIP Image Recognizer</h1>
      <p>Upload an image and get the top {{ top_k }} ImageNet predictions from CLIP.</p>
      <form method="POST" enctype="multipart/form-data">
        <label for="image">Choose an image:</label>
        <input type="file" id="image" name="image" accept="image/*" required>
        <div style="margin-top: 1rem;">
          <button type="submit">Classify image</button>
        </div>
        {% if error %}
        <div class="error">{{ error }}</div>
        {% endif %}
      </form>

      {% if predictions %}
      <div class="predictions">
        <h2>Top predictions</h2>
        <table>
          <thead>
            <tr><th>Rank</th><th>Label</th><th>Probability</th></tr>
          </thead>
          <tbody>
            {% for idx, (label, score) in enumerate(predictions, start=1) %}
            <tr>
              <td>{{ idx }}</td>
              <td>{{ label }}</td>
              <td>{{ '%.4f'|format(score) }}</td>
            </tr>
            {% endfor %}
          </tbody>
        </table>
      </div>
      {% endif %}
    </div>
  </body>
</html>
"""


def _extract_image(upload) -> Image.Image:
    """Load an uploaded file into a PIL image."""

    content = upload.read()
    if not content:
        raise ValueError("Uploaded file is empty")
    return Image.open(BytesIO(content)).convert("RGB")


def classify_image(upload, top_k: Optional[int] = None):
    top_k = top_k or DEFAULT_TOP_K
    image = _extract_image(upload)
    predictions = rank_labels_from_image(
        image=image,
        labels=LABELS,
        device=DEVICE,
        model=MODEL,
        preprocess=PREPROCESS,
    )
    return predictions[:top_k]


@app.route("/", methods=["GET", "POST"])
def index():
    predictions = None
    error = None

    if request.method == "POST":
        upload = request.files.get("image")
        if not upload or upload.filename == "":
            error = "Please choose an image to upload."
        else:
            try:
                predictions = classify_image(upload)
            except Exception as exc:  # noqa: BLE001 - surface classification errors
                error = f"Failed to classify image: {exc}"

    return render_template_string(
        PAGE_TEMPLATE,
        predictions=predictions,
        error=error,
        top_k=DEFAULT_TOP_K,
    )


if __name__ == "__main__":
    app.run(debug=True)
