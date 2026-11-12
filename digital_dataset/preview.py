"""Draw a contact sheet of generated digits (used for the figure in the README).

Each script gets a few rows, one per font, with the digits 0-9 side by side.
Fonts are picked at regular steps through the sorted font list, so the sheet
is the same every time it is generated.

    python digital_dataset/preview.py --out digital_dataset/assets/samples.png
"""
import argparse

from PIL import Image, ImageDraw

from generate_digital import SCRIPTS, covers_digits, list_fonts, render_font_digits


def pick(items, count):
    if len(items) <= count:
        return items
    step = len(items) / count
    return [items[int(i * step)] for i in range(count)]


def build_sheet(font_root, size, rows_per_script, fill, scale, scripts):
    gap = 2
    label_w = 96
    rows = []
    for script in scripts:
        digits, folders = SCRIPTS[script]
        fonts = [f for f in list_fonts(font_root, folders) if covers_digits(f, digits)]
        for path in pick(fonts, rows_per_script):
            imgs = render_font_digits(digits, path, size, fill)
            if imgs is not None:
                rows.append((script, imgs))

    cell = size * scale
    width = label_w + 10 * (cell + gap)
    sheet = Image.new("L", (width, len(rows) * (cell + gap)), 40)
    draw = ImageDraw.Draw(sheet)
    previous = None
    for r, (script, imgs) in enumerate(rows):
        y = r * (cell + gap)
        if script != previous:
            draw.text((6, y + cell // 2 - 5), script, fill=230)
            previous = script
        for c, img in enumerate(imgs):
            sheet.paste(img.resize((cell, cell), Image.NEAREST), (label_w + c * (cell + gap), y))
    return sheet


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fonts", default="digital_dataset/fonts")
    p.add_argument("--out", default="digital_dataset/assets/samples.png")
    p.add_argument("--size", type=int, default=32)
    p.add_argument("--rows", type=int, default=2, help="fonts shown per script")
    p.add_argument("--fill", type=float, default=0.7)
    p.add_argument("--scale", type=int, default=2, help="enlargement factor for viewing")
    p.add_argument("--scripts", nargs="+", default=list(SCRIPTS), choices=list(SCRIPTS))
    a = p.parse_args()
    build_sheet(a.fonts, a.size, a.rows, a.fill, a.scale, a.scripts).save(a.out)
    print("saved", a.out)
