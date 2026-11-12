"""Stage 2: train the language classifier on images restored by the frozen U-Net.

    python -m scripts.train_language --unet checkpoints/mra_unet.keras --out checkpoints/language.keras
"""
import argparse
from pathlib import Path

import keras
import numpy as np
from sklearn.model_selection import train_test_split

from mranet.data import LANGUAGE_NAMES, handwritten_dataset, load_language
from mranet.models import build_language_classifier, with_frozen_unet
from mranet.utils import plot_history, set_seed


def collect(split, limit, seed):
    """Stack one split of all languages; the label is the index of the language."""
    images, labels = [], []
    rng = np.random.default_rng(seed)
    for index, name in enumerate(LANGUAGE_NAMES):
        x = split[name]
        keep = rng.permutation(len(x))[:limit] if limit else range(len(x))
        images += [x[i] for i in keep]
        labels += [index] * len(keep)
    return images, np.array(labels, dtype=np.int64)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--unet", default="checkpoints/mra_unet.keras")
    p.add_argument("--data", default="datasets/handwritten")
    p.add_argument("--out", default="checkpoints/language.keras")
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--limit-per-language", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    set_seed(args.seed)

    train, test = {}, {}
    for name in LANGUAGE_NAMES:
        x_train, _, x_test, _ = load_language(name, args.data)
        train[name], test[name] = x_train, x_test
    x_all, y_all = collect(train, args.limit_per_language, args.seed)
    idx_train, idx_val = train_test_split(np.arange(len(x_all)), test_size=0.2, random_state=args.seed,
                                          stratify=y_all)
    x_test, y_test = collect(test, args.limit_per_language, args.seed)

    def ds(images, labels, shuffle):
        return handwritten_dataset(images, labels, args.size, args.batch_size, shuffle=shuffle, seed=args.seed)

    train_ds = ds([x_all[i] for i in idx_train], y_all[idx_train], True)
    val_ds = ds([x_all[i] for i in idx_val], y_all[idx_val], False)
    test_ds = ds(x_test, y_test, False)

    unet = keras.saving.load_model(args.unet)
    classifier = build_language_classifier((args.size, args.size, 1), len(LANGUAGE_NAMES))
    model = with_frozen_unet(unet, classifier, "restore_then_classify_language")
    model.compile(optimizer=keras.optimizers.Adam(args.lr), loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs)
    print("test:", model.evaluate(test_ds, verbose=0, return_dict=True))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    classifier.save(args.out)  # the U-Net is saved separately and chained again in the next stage
    plot_history(history, Path(args.out).with_suffix(".history.png"))
    print("saved", args.out)


if __name__ == "__main__":
    main()
