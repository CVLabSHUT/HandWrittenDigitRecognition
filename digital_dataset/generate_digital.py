"""Generate the synthetic DIGITal dataset from font files.

Every digit is drawn large, cropped to the pixels that were actually inked and
rescaled so that the largest digit of the font fills a fixed square frame; the
frame is placed in the middle of the image. The result does not depend on the metrics
(ascent, descent, side bearings, default em size) of a particular font, so two
fonts rendered "at the same size" end up with digits of the same size and
position. The old generator used the requested font size as the canvas size
and trusted the font metrics for centring, which gave digits that were clipped,
tiny or off-centre depending on the font.

Example:
    python digital_dataset/generate_digital.py --out generated \
        --scripts latin persian_urdu --sizes 32 64 --fill 0.7
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont

# Digits 0-9 of every supported script and the folders (inside --fonts) that
# hold fonts covering them.
SCRIPTS = {
    "latin": ("0123456789", ["English", "ARDIS"]),
    "arabic": ("".join(chr(0x0660 + i) for i in range(10)), ["Arabic"]),
    "persian_urdu": ("".join(chr(0x06F0 + i) for i in range(10)), ["Farsi_Urdu"]),
    "chinese": ("零一二三四五六七八九", ["Chinese"]),
    "gujarati": ("".join(chr(0x0AE6 + i) for i in range(10)), ["Gujarati"]),
    "gurmukhi": ("".join(chr(0x0A66 + i) for i in range(10)), ["Gurmukhi"]),
    "bangla": ("".join(chr(0x09E6 + i) for i in range(10)), ["ISIBangla_Banglalekha"]),
    "kannada": ("".join(chr(0x0CE6 + i) for i in range(10)), ["Kannada"]),
    "tibetan": ("".join(chr(0x0F20 + i) for i in range(10)), ["Tibetan"]),
}
FONT_SUFFIXES = {".ttf", ".otf", ".ttc"}


def list_fonts(font_root, folders):
    paths = []
    for folder in folders:
        folder_path = Path(font_root) / folder
        if folder_path.is_dir():
            paths += [p for p in folder_path.iterdir()
                      if p.suffix.lower() in FONT_SUFFIXES and not p.name.startswith("._")]
    return sorted(paths)


def covers_digits(font_path, digits):
    """True if the font has a glyph for every digit in `digits`."""
    try:
        cmap = TTFont(str(font_path), fontNumber=0, lazy=True).getBestCmap()
    except Exception:
        return False
    return cmap is not None and all(ord(d) in cmap for d in digits)


def render_ink(char, font_path, px):
    """Draw `char` on a large canvas and return it cropped to the inked pixels.

    The glyph is anchored at the middle of the canvas, so font metrics only
    decide where the ink lands inside the canvas, never how it is cropped.
    Returns None if nothing was drawn or the glyph does not fit.
    """
    side = px * 4
    canvas = Image.new("L", (side, side), 0)
    font = ImageFont.truetype(str(font_path), px)
    ImageDraw.Draw(canvas).text((side // 2, side // 2), char, fill=255, font=font, anchor="mm")
    box = canvas.getbbox()
    if box is None:
        return None
    left, top, right, bottom = box
    if left == 0 or top == 0 or right == side or bottom == side:
        return None  # clipped by the canvas border
    return canvas.crop(box)


def looks_like_box(ink):
    """True for a hollow rectangle (also with rounded or slanted corners), the usual placeholder for a missing glyph."""
    arr = np.asarray(ink) > 40
    ys, xs = np.nonzero(arr)
    if len(ys) == 0:
        return False
    arr = arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]  # tight box around the clearly inked pixels
    h, w = arr.shape
    if h < 12 or w < 12:
        return False
    band = max(2, round(0.04 * min(h, w)))
    edges = (arr[:band].mean(), arr[-band:].mean(), arr[:, :band].mean(), arr[:, -band:].mean())
    interior = arr[h // 4:3 * h // 4, w // 4:3 * w // 4].mean()  # the border of a box can be thick
    return min(edges) > 0.55 and interior < 0.12


def looks_like_block(ink):
    """True for a solid, roughly square-ish block that fills its bounding box (no digit does, except thin bars)."""
    arr = np.asarray(ink) > 40
    ys, xs = np.nonzero(arr)
    if len(ys) == 0:
        return False
    arr = arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = arr.shape
    return arr.mean() > 0.75 and min(h, w) > 0.2 * max(h, w)


def nearly_identical(a, b):
    """Two ink images of the same size that differ in fewer than 2% of the grey value range."""
    if a.size != b.size:
        return False
    diff = np.abs(np.asarray(a, dtype=np.int16) - np.asarray(b, dtype=np.int16))
    return diff.mean() / 255 < 0.02


def has_placeholder(inks, font_path, px, check_shapes=False):
    """True if one of the digit images is a placeholder rather than a real glyph.

    A font can list a digit in its character table and still draw a "missing
    character" box for it, so the character table alone is not enough. Three
    signs are checked: the image equals what the font draws for a character it
    certainly does not have, it is a hollow box or a solid block, or two digits
    look the same.
    """
    missing = render_ink("􏿿", font_path, px)
    if missing is not None and any(nearly_identical(ink, missing) for ink in inks):
        return True
    if check_shapes:  # only used where such boxes were found: Arabic-script fonts
        if any(looks_like_box(ink) for ink in inks[1:]):  # zero is a dot or a ring in several scripts
            return True
        if any(looks_like_block(ink) for ink in inks[2:]):  # one is a bar in several scripts
            return True
    return any(nearly_identical(inks[i], inks[j]) for i in range(len(inks)) for j in range(i + 1, len(inks)))


def place(ink, size, scale, center="bbox"):
    """Resize `ink` by `scale` and put it on a size x size canvas, centred."""
    w, h = ink.size
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    ink = ink.resize((new_w, new_h), Image.LANCZOS)

    if center == "mass":
        arr = np.asarray(ink, dtype=np.float64)
        ys, xs = np.mgrid[0:new_h, 0:new_w]
        cx, cy = (arr * xs).sum() / arr.sum(), (arr * ys).sum() / arr.sum()
        x0 = min(max(round(size / 2 - cx), 0), size - new_w)
        y0 = min(max(round(size / 2 - cy), 0), size - new_h)
    else:
        x0, y0 = (size - new_w) // 2, (size - new_h) // 2

    out = Image.new("L", (size, size), 0)
    out.paste(ink, (x0, y0))
    return out


def render_font_digits(digits, font_path, size, fill, center="bbox", normalize="font"):
    """Render all digits of one font. Returns a list of images, or None if the font fails.

    normalize="font": one scale for the whole set, chosen so that the largest digit
        fills the frame. Proportions inside a font survive (the Arabic zero stays a
        small dot, a "1" stays narrow).
    normalize="glyph": every digit is scaled on its own to fill the frame.
    """
    frame = max(1, round(size * fill))
    # Very tall glyphs (Tibetan, ornate fonts) can be clipped at a large render size,
    # so fall back to smaller ones. All digits of a font share the same render size.
    for px in (max(256, 2 * size), 160, 80):
        inks = [render_ink(d, font_path, px) for d in digits]
        if all(ink is not None for ink in inks):
            break
    else:
        return None
    arabic_script = 0x0600 <= ord(digits[0]) <= 0x06FF
    if has_placeholder(inks, font_path, px, check_shapes=arabic_script):
        return None

    if normalize == "font":
        scale = frame / max(max(ink.size) for ink in inks)
        return [place(ink, size, scale, center) for ink in inks]
    return [place(ink, size, frame / max(ink.size), center) for ink in inks]


def generate(font_root, out_dir, scripts, sizes, fills, center, fmt, normalize="font"):
    out_dir = Path(out_dir)
    for script in scripts:
        digits, folders = SCRIPTS[script]
        fonts = list_fonts(font_root, folders)
        usable, skipped = [], 0
        for path in fonts:
            if covers_digits(path, digits):
                usable.append(path)
            else:
                skipped += 1
        print(f"{script}: {len(usable)} usable fonts, {skipped} skipped (missing digit glyphs)")

        for size in sizes:
            images, labels, font_ids, fill_ids = [], [], [], []
            rows = []
            for font_id, path in enumerate(usable):
                for fill in fills:
                    batch = render_font_digits(digits, path, size, fill, center, normalize)
                    if batch is None:
                        print(f"  skipping {path.name}: digits could not be drawn")
                        break
                    for label, img in enumerate(batch):
                        rows.append((font_id, path.name, label, fill))
                        if fmt == "png":
                            target = out_dir / script / f"{size}x{size}" / str(label)
                            target.mkdir(parents=True, exist_ok=True)
                            img.save(target / f"f{font_id:04d}_fill{int(fill * 100)}.png")
                        else:
                            images.append(np.asarray(img, dtype=np.uint8))
                            labels.append(label)
                            font_ids.append(font_id)
                            fill_ids.append(int(fill * 100))

            folder = out_dir / script / f"{size}x{size}"
            folder.mkdir(parents=True, exist_ok=True)
            if fmt == "npz":
                np.savez_compressed(folder / "digits.npz", x=np.stack(images), y=np.array(labels),
                                    font_id=np.array(font_ids), fill_percent=np.array(fill_ids))
            with open(folder / "manifest.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["font_id", "font_file", "digit", "fill"])
                writer.writerows(rows)
            print(f"  {size}x{size}: {len(rows)} images")


def parse_args(argv):
    p = argparse.ArgumentParser(description="Generate synthetic digit images from fonts.")
    p.add_argument("--fonts", default="digital_dataset/fonts", help="folder with one sub-folder of fonts per script")
    p.add_argument("--out", default="generated", help="output folder")
    p.add_argument("--scripts", nargs="+", default=list(SCRIPTS), choices=list(SCRIPTS))
    p.add_argument("--sizes", nargs="+", type=int, default=[32], help="image side lengths in pixels")
    p.add_argument("--fill", nargs="+", type=float, default=[0.7],
                   help="frame side as a fraction of the image side (one image per value and font)")
    p.add_argument("--center", choices=["bbox", "mass"], default="bbox",
                   help="centre the ink bounding box (default) or the centre of mass")
    p.add_argument("--normalize", choices=["font", "glyph"], default="font",
                   help="scale each font's digits together (default) or every digit on its own")
    p.add_argument("--format", choices=["png", "npz"], default="png", dest="fmt")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    generate(args.fonts, args.out, args.scripts, args.sizes, args.fill, args.center, args.fmt, args.normalize)
