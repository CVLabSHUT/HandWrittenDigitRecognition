"""Build the upload package of the DIGITal dataset (Zenodo, Hugging Face, ...).

The package follows the paper: 100,000 images of 128 x 128, ten groups of 10,000.
Only images and tables are written; the font files are not copied.

    python digital_dataset/make_release.py --out ../DIGITal_release

Creates in the output folder:
    DIGITal_npz.zip     digits.npz + manifest.csv for every group
    DIGITal_png.zip     the same images as PNG files, one folder per digit
    splits.csv          train / validation / test split by font, per group
    fonts_used.csv      the fonts behind the images with the license information found in the files
    README.md           dataset card
    CHECKSUMS.sha256    SHA-256 of every file above
"""
import argparse
import csv
import hashlib
import shutil
import zipfile
from pathlib import Path

import numpy as np

from generate_paper_dataset import GROUPS, write_group

HERE = Path(__file__).parent
TITLES = {"arabic": "Arabic", "ardis": "ARDIS (Latin digits)", "chinese": "Chinese",
          "farsi_urdu": "Farsi and Urdu", "gujarati": "Gujarati", "bangla": "ISI Bangla and BanglaLekha",
          "kannada": "Kannada", "tibetan": "Tibetan", "english": "English (Latin digits)",
          "gurmukhi": "Gurmukhi"}


def zip_folder(folder, target):
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                z.write(path, path.relative_to(folder.parent))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_license_report():
    report = {}
    path = HERE / "font_license_report.csv"
    if path.exists():
        with open(path, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                report[(row["font_folder"], row["file"])] = row
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fonts", default="digital_dataset/fonts")
    p.add_argument("--out", default="DIGITal_release")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} already exists; remove it or choose another folder")
    out.mkdir(parents=True)
    report = load_license_report()
    rng = np.random.default_rng(args.seed)

    splits, fonts_used, table = [], [], []
    for number, group in enumerate(GROUPS):
        for fmt in ("npz", "png"):
            records, n_fonts = write_group(out / f"data_{fmt}", group, fmt, 1000, 0.7, 15.0, 10,
                                           args.seed + number, args.fonts)
        font_files = {}
        for font_id, file_name, *_ in records:
            font_files[font_id] = file_name
        order = rng.permutation(len(font_files))
        n_test = n_val = max(1, round(0.1 * len(order)))
        folders = GROUPS[group][1]
        for rank, font_id in enumerate(order):
            role = "test" if rank < n_test else "validation" if rank < n_test + n_val else "train"
            file_name = font_files[int(font_id)]
            info = next((report[(f, file_name)] for f in folders if (f, file_name) in report), {})
            splits.append([group, int(font_id), file_name, role])
            fonts_used.append([group, int(font_id), info.get("family", ""), info.get("category", "not audited"),
                               info.get("license_url", ""), info.get("copyright", "")])
        table.append(f"| {TITLES[group]} | `{group}` | {n_fonts} | 10,000 |")

    splits.sort(key=lambda r: (r[0], r[1]))
    fonts_used.sort(key=lambda r: (r[0], r[1]))
    with open(out / "splits.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["group", "font_id", "font_file", "split"])
        w.writerows(splits)
    with open(out / "fonts_used.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["group", "font_id", "family", "license_category_in_file", "license_url_in_file",
                    "copyright_in_file"])
        w.writerows(fonts_used)

    for fmt in ("npz", "png"):
        zip_folder(out / f"data_{fmt}", out / f"DIGITal_{fmt}.zip")
        shutil.rmtree(out / f"data_{fmt}")
    card = (HERE / "release_card.md").read_text(encoding="utf-8")
    (out / "README.md").write_text(card.replace("{{GROUP_TABLE}}", "\n".join(table)), encoding="utf-8")

    with open(out / "CHECKSUMS.sha256", "w") as f:
        for path in sorted(out.iterdir()):
            if path.is_file() and path.name != "CHECKSUMS.sha256":
                f.write(f"{sha256(path)}  {path.name}\n")
    print("release package written to", out)


if __name__ == "__main__":
    main()
