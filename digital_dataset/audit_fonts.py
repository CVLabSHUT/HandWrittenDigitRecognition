"""List the license information embedded in the font files.

Reads the name table (copyright, license description, license URL) and the OS/2
embedding flags of every font and sorts the fonts into rough classes:

    open         SIL OFL, Apache, Creative Commons, MIT, public domain and similar
    personal     free for personal / non-commercial use only
    restricted   proprietary wording or embedding restricted by the font itself
    unknown      no license text in the file

This is evidence from the files, not legal advice: embedded text can be missing
or wrong, so "open" fonts should still be spot-checked at their source.

    python digital_dataset/audit_fonts.py --out digital_dataset/font_license_report.csv
"""
import argparse
import csv
import re
from collections import Counter
from pathlib import Path

from fontTools.ttLib import TTFont

OPEN = [
    (r"open font license|scripts\.sil\.org/ofl|openfontlicense|\bofl\b", "SIL OFL"),
    (r"apache", "Apache"),
    (r"creativecommons\.org|creative commons|\bcc0\b|\bcc[- ]by\b", "Creative Commons"),
    (r"public domain|publicdomain", "Public domain"),
    (r"ubuntu font licen[sc]e", "Ubuntu Font License"),
    (r"gnu general public license|\bgpl\b", "GPL (check font exception)"),
    (r"permission is hereby granted|\bmit license\b", "MIT"),
    (r"bitstream vera|dejavu", "Bitstream Vera / DejaVu"),
]
PERSONAL = r"personal use|non[- ]?commercial|not for commercial|free for personal|for personal and non"
RESTRICTED = (r"all rights reserved|commercial use|licen[sc]e agreement|eula|may not be|not be (copied|"
              r"modified|redistributed)|trademark of|monotype|linotype|adobe systems|microsoft corporation|"
              r"itc |agfa|bitstream inc|fontfont|letraset")


def name(font, ids):
    for i in ids:
        text = font["name"].getDebugName(i)
        if text:
            text = ''.join(ch if ch.isprintable() else ' ' for ch in text)  # control characters break the CSV
            return re.sub(r"\s+", " ", text).strip()
    return ""


def classify(license_text, url, copyright_text, fs_type):
    blob = f"{license_text} {url} {copyright_text}".lower()
    for pattern, label in OPEN:
        if re.search(pattern, blob):
            return "open", label
    if re.search(PERSONAL, blob):
        return "personal", "personal / non-commercial wording"
    if fs_type in (2,):
        return "restricted", "embedding flag: restricted license"
    if re.search(RESTRICTED, blob):
        return "restricted", "proprietary or restrictive wording"
    if not license_text and not url:
        return "unknown", "no license text in the file" + (f" (copyright: {copyright_text[:60]})" if copyright_text else "")
    return "unknown", "license text not recognised"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fonts", default="digital_dataset/fonts")
    p.add_argument("--out", default="digital_dataset/font_license_report.csv")
    args = p.parse_args()

    rows = []
    for path in sorted(Path(args.fonts).rglob("*")):
        if path.suffix.lower() not in {".ttf", ".otf", ".ttc"} or path.name.startswith("._"):
            continue
        try:
            font = TTFont(str(path), fontNumber=0, lazy=True)
            copyright_text = name(font, [0])
            license_text = name(font, [13])
            url = name(font, [14])
            fs_type = font["OS/2"].fsType if "OS/2" in font else None
            category, why = classify(license_text, url, copyright_text, fs_type)
            rows.append([path.parent.name, path.name, name(font, [16, 1]), copyright_text[:200],
                         license_text[:300], url, name(font, [9]), fs_type, category, why])
        except Exception as e:
            rows.append([path.parent.name, path.name, "", "", "", "", "", None, "unknown", f"unreadable: {e}"])

    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["font_folder", "file", "family", "copyright", "license_text", "license_url",
                         "designer", "os2_fsType", "category", "reason"])
        writer.writerows(rows)

    print(f"{len(rows)} fonts -> {args.out}\n")
    by_folder = {}
    for r in rows:
        by_folder.setdefault(r[0], Counter())[r[8]] += 1
    print(f"{'folder':26s} {'open':>5s} {'personal':>9s} {'restricted':>11s} {'unknown':>8s}")
    for folder, c in sorted(by_folder.items()):
        print(f"{folder:26s} {c['open']:5d} {c['personal']:9d} {c['restricted']:11d} {c['unknown']:8d}")
    total = Counter(r[8] for r in rows)
    print(f"{'TOTAL':26s} {total['open']:5d} {total['personal']:9d} {total['restricted']:11d} {total['unknown']:8d}")


if __name__ == "__main__":
    main()
