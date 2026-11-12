"""Stage 1: train the MRA U-Net to restore degraded printed digits.

The printed digits come from the DIGITal generator, for example:
    python digital_dataset/generate_digital.py --sizes 128 --format npz --out generated

Then:
    python -m scripts.train_restoration --printed generated --out checkpoints/mra_unet.keras
"""
import argparse
from pathlib import Path

import keras
import numpy as np

from mranet.data import restoration_dataset
from mranet.models import build_mra_unet
from mranet.utils import plot_history, set_seed


def load_printed(root, size, limit=None, seed=0):
    files = sorted(Path(root).glob(f"*/{size}x{size}/digits.npz"))
    if not files:
        raise SystemExit(f"no digits.npz for size {size} found under {root}")
    images = np.concatenate([np.load(f)["x"] for f in files])
    if limit and limit < len(images):
        images = images[np.random.default_rng(seed).choice(len(images), limit, replace=False)]
    return images


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--printed", default="generated", help="output folder of generate_digital.py")
    p.add_argument("--out", default="checkpoints/mra_unet.keras")
    p.add_argument("--size", type=int, default=128, help="image side, divisible by 16")
    p.add_argument("--degrade-size", type=int, default=28,
                   help="side of the low-resolution image used to degrade the inputs")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--limit", type=int, default=None, help="use only this many printed images")
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    set_seed(args.seed)

    images = load_printed(args.printed, args.size, args.limit, args.seed)
    n_val = max(1, int(len(images) * args.val_fraction))
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(images))
    val_images, train_images = images[order[:n_val]], images[order[n_val:]]
    print(f"{len(train_images)} training and {n_val} validation images")

    train_ds = restoration_dataset(train_images, args.degrade_size, args.batch_size, seed=args.seed)
    val_ds = restoration_dataset(val_images, args.degrade_size, args.batch_size, shuffle=False)

    model = build_mra_unet((args.size, args.size, 1))
    model.compile(optimizer=keras.optimizers.Adam(args.lr), loss="binary_crossentropy",
                  metrics=["mae"])
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.out)
    plot_history(history, Path(args.out).with_suffix(".history.png"))
    print("saved", args.out)


if __name__ == "__main__":
    main()
