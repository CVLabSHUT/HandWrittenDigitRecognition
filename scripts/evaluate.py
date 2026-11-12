"""Evaluate a trained digit model on the test split of a language.

    python -m scripts.evaluate --language Persian --model checkpoints/digit_Persian.keras
"""
import argparse
from pathlib import Path

import keras
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

from mranet.data import LANGUAGES, handwritten_dataset, load_language


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--language", required=True, choices=list(LANGUAGES))
    p.add_argument("--model", required=True)
    p.add_argument("--data", default="datasets/handwritten")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--out", default=None, help="confusion matrix image (default: next to the model)")
    args = p.parse_args()

    model = keras.saving.load_model(args.model)
    size = model.input_shape[1]
    _, _, x_test, y_test = load_language(args.language, args.data)
    probs = model.predict(handwritten_dataset(x_test, y_test, size, args.batch_size), verbose=0)
    pred = probs.argmax(axis=1)

    print(f"{args.language}: accuracy {np.mean(pred == y_test):.4f} on {len(y_test)} test images")
    print(classification_report(y_test, pred, digits=4))

    out = Path(args.out or Path(args.model).with_suffix(".confusion.png"))
    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay(confusion_matrix(y_test, pred)).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{args.language}: test confusion matrix")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print("saved", out)


if __name__ == "__main__":
    main()
