# Handwritten digit datasets

The twelve handwritten datasets used in the paper, stored in `handwritten/`. Digits
are light on a dark background in every file. Pixel values are either 0-255 integers
or 0-1 floats; `mranet.data.load_language` converts everything to floats in [0, 1].

| Language | File | Train | Test | Native size | Classes | Probable original source |
|---|---|---:|---:|---|---:|---|
| Arabic | `Arabic.pkl` | 60,499 | 10,000 | 28 x 28 | 10 | MADBase |
| ARDIS | `ARDIS.npz` | 6,600 | 1,000 | 28 x 28 | 10 | ARDIS (Swedish historical digits) |
| BanglaLekha | `BanglaLekha.npz` | 15,798 | 3,950 | 28 x 28 | 10 | BanglaLekha-Isolated |
| Chinese | `Chinese.pkl` | 58,946 | 15,000 | 64 x 64 | 15 | Chinese MNIST |
| English | `English.pkl` | 19,998 | 1,500 | varies (about 30-150 px) | 10 | not identified |
| Gujarati | `Gujarati.pkl` | 5,600 | 2,400 | 256 x 256 | 10 | not identified |
| Gurmukhi | `Gurmukhi.npz` | 1,000 | 178 | 32 x 32 | 10 | not identified |
| ISI Bangla | `ISI_Bangla.npz` | 19,392 | 4,000 | 28 x 28 | 10 | ISI handwritten Bangla numerals |
| Kannada | `Kannada.npz` | 60,000 | 20,240 | 28 x 28 | 10 | Kannada-MNIST (test = test set + Dig-MNIST) |
| Persian | `Persian.npz` | 60,000 | 20,000 | 28 x 28 | 10 | HODA |
| Tibetan | `Tibetan.npz` | 14,214 | 3,554 | 28 x 28 | 10 | not identified |
| Urdu | `Urdu.npz` | 6,606 | 1,414 | 28 x 28 | 10 | not identified |

The last column is an inference from file names and sizes, **not a record from the
authors**; it has to be confirmed before release. Each dataset has its own license
and citation requirements. Please cite the original sources when you use them and
make sure that redistributing the files here is permitted.

## File formats

- `.npz`: arrays `X_train`, `y_train`, `X_test`, `y_test`.
- `.pkl`: a bz2-compressed pickle of the tuple `(X_train, y_train, X_test, y_test)`.
- English images have different sizes, so `English.pkl` holds an array of objects.
  They are padded to squares before resizing.
- Chinese has 15 classes: the ten digits plus larger denominations.

## Loading

```python
from mranet.data import load_language

x_train, y_train, x_test, y_test = load_language("Persian")   # lists of float32 images, int64 labels
```
