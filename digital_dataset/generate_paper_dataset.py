"""Generate the DIGITal dataset as it is described in the paper.

    100,000 images of 128 x 128 pixels: ten groups of 10,000 images, 1,000 per digit.

The ten groups are those of Table 1 of the paper. Farsi and Urdu share one group,
and so do ISI Bangla and BanglaLekha, because their digits look the same. ARDIS and
English are separate groups, each with its own fonts.

Every font gives one clean image per digit (see generate_digital.py for how a digit is
placed in the frame). Groups with fewer than 1,000 fonts are filled up by reusing the
fonts with a random horizontal shift and rotation, which is the augmentation the paper
uses to balance languages with few fonts. The paper does not give the size of the shift
or the angle, so they are options here.

    python digital_dataset/generate_paper_dataset.py --out generated
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops

from generate_digital import SCRIPTS, covers_digits, list_fonts, render_font_digits

SIZE = 128

# group -> (script of generate_digital.SCRIPTS, font folders)
GROUPS = {
    "arabic": ("arabic", ["Arabic"]),
    "ardis": ("latin", ["ARDIS"]),
    "chinese": ("chinese", ["Chinese"]),
    "farsi_urdu": ("persian_urdu", ["Farsi_Urdu"]),
    "gujarati": ("gujarati", ["Gujarati"]),
    "bangla": ("bangla", ["ISIBangla_Banglalekha"]),
    "kannada": ("kannada", ["Kannada"]),
    "tibetan": ("tibetan", ["Tibetan"]),
    "english": ("latin", ["English"]),
    "gurmukhi": ("gurmukhi", ["Gurmukhi"]),
}


def augment(img, angle, shift_x):
    """Rotate around the centre, then move horizontally (nothing wraps around: the frame leaves a margin)."""
    img = img.rotate(angle, resample=Image.BICUBIC, fillcolor=0)
    return ImageChops.offset(img, shift_x, 0)


def usable_fonts(font_root, group, size, fill):
    script, folders = GROUPS[group]
    digits = SCRIPTS[script][0]
    fonts = []
    for path in list_fonts(font_root, folders):
        if covers_digits(path, digits):
            images = render_font_digits(digits, path, size, fill)
            if images is not None:
                fonts.append((path, images))
    return fonts


def build_group(font_root, group, per_digit, size, fill, max_rotation, max_shift, seed):
    fonts = usable_fonts(font_root, group, size, fill)
    rng = np.random.default_rng(seed)
    images, labels, records = [], [], []
    for digit in range(10):
        for i in range(per_digit):
            font_id, variant = i % len(fonts), i // len(fonts)
            img = fonts[font_id][1][digit]
            angle, shift_x = 0.0, 0
            if variant > 0:  # the first pass over the fonts is left untouched
                angle = float(rng.uniform(-max_rotation, max_rotation))
                shift_x = int(rng.integers(-max_shift, max_shift + 1))
                img = augment(img, angle, shift_x)
            images.append(np.asarray(img, dtype=np.uint8))
            labels.append(digit)
            records.append((font_id, fonts[font_id][0].name, digit, variant, round(angle, 2), shift_x))
    return np.stack(images), np.array(labels, dtype=np.int8), records, len(fonts)


def write_group(out, group, fmt, per_digit, fill, max_rotation, max_shift, seed, font_root):
    """Generate one group and write it to <out>/<group>/128x128/. Returns the manifest records."""
    x, y, records, n_fonts = build_group(font_root, group, per_digit, SIZE, fill, max_rotation, max_shift, seed)
    folder = Path(out) / group / f"{SIZE}x{SIZE}"
    folder.mkdir(parents=True, exist_ok=True)
    if fmt == "npz":
        np.savez_compressed(folder / "digits.npz", x=x, y=y,
                            font_id=np.array([r[0] for r in records], dtype=np.int16),
                            variant=np.array([r[3] for r in records], dtype=np.int16),
                            rotation=np.array([r[4] for r in records], dtype=np.float32),
                            shift_x=np.array([r[5] for r in records], dtype=np.int8))
    else:
        for index, (img, record) in enumerate(zip(x, records)):
            target = folder / str(record[2])
            target.mkdir(exist_ok=True)
            Image.fromarray(img).save(target / f"{index:05d}.png")
    with open(folder / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["index", "font_id", "font_file", "digit", "variant", "rotation_deg", "shift_x"])
        w.writerows([(i, *r) for i, r in enumerate(records)])
    print(f"{group}: {n_fonts} fonts -> {len(x)} images of {SIZE}x{SIZE}")
    return records, n_fonts


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fonts", default="digital_dataset/fonts")
    p.add_argument("--out", default="generated")
    p.add_argument("--groups", nargs="+", default=list(GROUPS), choices=list(GROUPS))
    p.add_argument("--per-digit", type=int, default=1000, help="images per digit and group")
    p.add_argument("--fill", type=float, default=0.7, help="frame side as a fraction of the image side")
    p.add_argument("--max-rotation", type=float, default=15.0, help="degrees")
    p.add_argument("--max-shift", type=int, default=10, help="pixels, horizontal")
    p.add_argument("--format", choices=["png", "npz"], default="npz", dest="fmt")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    for number, group in enumerate(args.groups):
        write_group(args.out, group, args.fmt, args.per_digit, args.fill, args.max_rotation, args.max_shift,
                    args.seed + number, args.fonts)


if __name__ == "__main__":
    main()
