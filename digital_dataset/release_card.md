# DIGITal: a synthetic multilingual digit dataset

DIGITal contains 100,000 printed digit images of 128 x 128 pixels in ten groups of digits (12
languages), rendered from more than 1,000 fonts. It was created for the paper *Advancing
Multilingual Handwritten Numeral Recognition with Attention-driven Transfer Learning* (IEEE Access,
2024), where printed digits are used to train a restoration network that is later transferred to
handwritten digits.

- Paper: https://ieeexplore.ieee.org/document/10474004
- Code and generator: https://github.com/CVLabSHUT/HandWrittenDigitRecognition
- License: CC BY-NC 4.0 (non-commercial research and education, attribution required)

## Contents

Ten groups of **10,000 images each, 1,000 per digit**, as in Table 1 and Table 3 of the paper.
Farsi and Urdu share a group, and so do ISI Bangla and BanglaLekha, because their digits look the
same.

| Group | Name in files | Fonts | Images |
|---|---|---:|---:|
{{GROUP_TABLE}}
| **Total** | | | **100,000** |

All images are **128 x 128** pixels, 8-bit grayscale, white digit (255) on a black (0) background.

Each font contributes one clean image per digit. Groups with few fonts are filled up to 1,000 images
per digit by reusing the fonts with a random horizontal shift (up to 10 pixels) and rotation (up to
15 degrees), the augmentation used in the paper to balance languages with few fonts. The
`variant` column tells the two apart: 0 is the clean render, larger values are augmented copies.

## Files

| File | Description |
|---|---|
| `DIGITal_npz.zip` | per group: `digits.npz` and `manifest.csv` |
| `DIGITal_png.zip` | the same images as PNG files, `<group>/128x128/<digit>/<index>.png` |
| `splits.csv` | split of every font into `train` (80%), `validation` (10%) and `test` (10%) |
| `fonts_used.csv` | the fonts behind the images and the license information found in the font files |
| `CHECKSUMS.sha256` | SHA-256 checksums |

`digits.npz` holds `x` (10000, 128, 128) uint8, `y` digit labels 0-9, `font_id`, `variant`,
`rotation` (degrees) and `shift_x` (pixels). `manifest.csv` has the same information per image with
the font file name.

```python
import numpy as np
data = np.load("farsi_urdu/128x128/digits.npz")
x, y, font_id = data["x"], data["y"], data["font_id"]
```

## Splits

Fonts, not images, are split into train, validation and test, so a model is evaluated on typefaces
it has never seen (including all augmented copies of them). Use `splits.csv` (columns `group,
font_id, font_file, split`) and the `font_id` of each image. The split uses a fixed seed (42).

## Low-quality inputs

The paper trains the restoration network on pairs of low-quality and high-quality images. This
release contains the high-quality images; the low-quality version is made by shrinking an image and
enlarging it back to 128 x 128. The code repository does this while loading the data
(`restoration_dataset` in `mranet/data.py`).

## How the images were made

Each digit is drawn large, cropped to its inked pixels and scaled into a fixed square frame (70% of
the image side) in the middle of the image. All ten digits of a font share one scale, chosen so that
the largest digit fills the frame. Results therefore do not depend on the size, ascent or side
bearings of a font, and proportions inside a font are kept (for example the Arabic zero stays a
small dot). The generator is in the code repository and the process is deterministic.

## Fonts and licensing

The dataset contains **images only; no font files are distributed**. The fonts come from many
sources and have different licenses. `fonts_used.csv` records the license information that is
embedded in each font file (for example SIL OFL or GPL, a copyright notice only, or nothing);
about one fifth of the fonts carry an open license. The images show rendered glyph shapes. If
you plan a use that is not research or education, check the licenses of the fonts you rely on.

## Intended use and limitations

Intended for pre-training, transfer-learning and font-robustness experiments, for building paired
clean and degraded digit images, and for low-resource scripts that have few handwritten samples.
These are printed digits, not handwriting: models should be fine-tuned and evaluated on real
handwritten data. Some fonts are decorative or very thin, and the number of fonts differs between
groups; groups with few fonts contain many augmented copies. The Chinese group has the ten digits
0-9 only. The sizes of the shift and rotation are not given in the paper; the values above are the
ones used to build this release.

## Citation

```bibtex
@article{fateh2024advancing,
  title   = {Advancing Multilingual Handwritten Numeral Recognition with Attention-driven Transfer Learning},
  author  = {Fateh, Amirreza and Tahmasbi Birgani, Reza and Fateh, Mansoor and Abolghasemi, Vahid},
  journal = {IEEE Access},
  year    = {2024}
}
```

Version 1.0.
