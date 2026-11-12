"""Stage 3: train the digit classifier of one language (or a baseline for comparison).

Proposed model: frozen U-Net -> frozen backbone of the language classifier -> new dense head.
    python -m scripts.train_digit --language Persian

Baselines trained on the raw images:
    python -m scripts.train_digit --language Persian --baseline lenet5
    python -m scripts.train_digit --language Persian --baseline vgg16
"""
import argparse
from pathlib import Path

import keras
import numpy as np
from sklearn.model_selection import train_test_split

from mranet.data import LANGUAGES, ROTATIONS, handwritten_dataset, load_language
from mranet.models import build_digit_classifier, build_lenet5, build_vgg16, with_frozen_unet
from mranet.utils import plot_history, set_seed


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--language", required=True, choices=list(LANGUAGES))
    p.add_argument("--baseline", choices=["lenet5", "vgg16"], default=None)
    p.add_argument("--unet", default="checkpoints/mra_unet.keras")
    p.add_argument("--language-model", default="checkpoints/language.keras")
    p.add_argument("--data", default="datasets/handwritten")
    p.add_argument("--out", default=None, help="default: checkpoints/digit_<language>[_<baseline>].keras")
    p.add_argument("--size", type=int, default=128, help="input side of the proposed model")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--no-rotation", action="store_true", help="turn off the rotation augmentation")
    p.add_argument("--limit", type=int, default=None, help="use only this many training images")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    set_seed(args.seed)

    n_classes = LANGUAGES[args.language][1]
    x_train, y_train, x_test, y_test = load_language(args.language, args.data)
    if args.limit:
        x_train, y_train = x_train[:args.limit], y_train[:args.limit]
    idx_train, idx_val = train_test_split(np.arange(len(x_train)), test_size=0.2, random_state=args.seed)
    angles = [] if args.no_rotation else ROTATIONS[args.language]

    # The baselines follow the usual 32x32 setup; the proposed model works on 128x128 images.
    size = {"lenet5": 32, "vgg16": 32}.get(args.baseline, args.size)

    def ds(indices, images, labels, rotate, shuffle):
        return handwritten_dataset([images[i] for i in indices], labels[indices], size, args.batch_size,
                                   angles=angles if rotate else (), shuffle=shuffle, seed=args.seed)

    train_ds = ds(idx_train, x_train, y_train, True, True)
    val_ds = ds(idx_val, x_train, y_train, False, False)
    test_ds = ds(np.arange(len(x_test)), x_test, y_test, False, False)

    if args.baseline == "lenet5":
        model = build_lenet5((size, size, 1), n_classes)
    elif args.baseline == "vgg16":
        model = build_vgg16((size, size, 1), n_classes)
    else:
        unet = keras.saving.load_model(args.unet)
        language_model = keras.saving.load_model(args.language_model)
        classifier = build_digit_classifier(language_model, n_classes)
        model = with_frozen_unet(unet, classifier, "restore_then_classify_digit")

    model.compile(optimizer=keras.optimizers.Adam(args.lr), loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs)
    print("test:", model.evaluate(test_ds, verbose=0, return_dict=True))

    out = args.out or f"checkpoints/digit_{args.language}{'_' + args.baseline if args.baseline else ''}.keras"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    model.save(out)
    plot_history(history, Path(out).with_suffix(".history.png"))
    print("saved", out)


if __name__ == "__main__":
    main()
