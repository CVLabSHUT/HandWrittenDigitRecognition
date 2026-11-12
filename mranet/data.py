"""Loading the handwritten digit datasets and building tf.data pipelines."""
import bz2
import pickle
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from scipy.ndimage import rotate

# language -> (file inside datasets/handwritten, number of digit classes)
# Chinese has 15 classes: the ten digits plus hundred, thousand, ten thousand,
# hundred million and so on, as in the original Chinese MNIST data.
LANGUAGES = {
    "Arabic": ("Arabic.pkl", 10),
    "ARDIS": ("ARDIS.npz", 10),
    "BanglaLekha": ("BanglaLekha.npz", 10),
    "Chinese": ("Chinese.pkl", 15),
    "English": ("English.pkl", 10),
    "Gujarati": ("Gujarati.pkl", 10),
    "Gurmukhi": ("Gurmukhi.npz", 10),
    "ISI_Bangla": ("ISI_Bangla.npz", 10),
    "Kannada": ("Kannada.npz", 10),
    "Persian": ("Persian.npz", 10),
    "Tibetan": ("Tibetan.npz", 10),
    "Urdu": ("Urdu.npz", 10),
}
# The position of a language in this list is its class index in the language classifier.
LANGUAGE_NAMES = list(LANGUAGES)

# Rotation angles (degrees) used to augment the training set of each language.
ROTATIONS = {
    "Arabic": [-35, -30, -25, -20, -15, -10, -5, 5, 10, 15, 20, 25, 30, 35],
    "ARDIS": [-30, -15, 15, 30],
    "BanglaLekha": [-30, 30],
    "English": [-30, 30],
    "Chinese": [30],
    "Gurmukhi": [-35, -30, -25, -20, -15, -10, -5, 5, 10, 15, 20, 25, 30, 35],
    "Persian": [-30, 30],
    "ISI_Bangla": [-30, 30],
    "Kannada": [30],
    "Tibetan": [-30, 30],
    "Urdu": [-30, -20, -10, 15, 30],
    "Gujarati": [-30, -20, -10, 10, 20, 30],
}


def _to_unit_square(img):
    """Scale pixel values to [0, 1] and pad the image with background to a square."""
    img = np.asarray(img, dtype=np.float32)
    if img.max() > 1.5:
        img = img / 255.0
    h, w = img.shape
    if h != w:
        side = max(h, w)
        padded = np.zeros((side, side), dtype=np.float32)
        top, left = (side - h) // 2, (side - w) // 2
        padded[top:top + h, left:left + w] = img
        img = padded
    return np.clip(img, 0.0, 1.0)


def load_language(name, root="datasets/handwritten"):
    """Return (x_train, y_train, x_test, y_test) for one language.

    The images come back as a list of square float32 arrays in [0, 1] with the
    native resolution of the dataset (28x28 for most, but 64x64 for Chinese and
    256x256 for Gujarati). Digits are light on a dark background in all sets.
    """
    file_name, _ = LANGUAGES[name]
    path = Path(root) / file_name
    if path.suffix == ".pkl":
        with bz2.BZ2File(path, "rb") as f:
            x_train, y_train, x_test, y_test = pickle.load(f)
    else:
        data = np.load(path, allow_pickle=True)
        x_train, y_train, x_test, y_test = (data[k] for k in ("X_train", "y_train", "X_test", "y_test"))
    return (
        [_to_unit_square(i) for i in x_train], np.asarray(y_train, dtype=np.int64),
        [_to_unit_square(i) for i in x_test], np.asarray(y_test, dtype=np.int64),
    )


def resize(img, size):
    interpolation = cv2.INTER_AREA if size < img.shape[0] else cv2.INTER_CUBIC
    return np.clip(cv2.resize(img, (size, size), interpolation=interpolation), 0.0, 1.0)


def handwritten_dataset(images, labels, size, batch_size, angles=(), shuffle=False, seed=0):
    """tf.data pipeline yielding (image, label) batches of size x size x 1 images.

    With `angles`, each image is rotated by one angle drawn at random from
    [0, *angles] every time it is visited. Rotation happens at the native
    resolution; resizing to `size` comes last.
    """
    choices = [0, *angles]
    epoch = [0]

    def generator():
        rng = np.random.default_rng(seed + epoch[0])
        epoch[0] += 1
        order = rng.permutation(len(images)) if shuffle else np.arange(len(images))
        for i in order:
            img = images[i]
            angle = choices[rng.integers(len(choices))]
            if angle:
                img = rotate(img, angle, reshape=False, order=1)
            yield resize(img, size)[..., None], labels[i]

    spec = (tf.TensorSpec((size, size, 1), tf.float32), tf.TensorSpec((), tf.int64))
    ds = tf.data.Dataset.from_generator(generator, output_signature=spec)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def restoration_dataset(clean, low_size, batch_size, shuffle=True, seed=0):
    """Pairs (degraded, clean) for training the restoration U-Net.

    `clean` is a uint8 array (N, size, size) of printed digits. The degraded
    input is made by shrinking the image to low_size x low_size and enlarging it
    back, which is how low-quality handwritten images look after upsampling.
    """
    size = clean.shape[1]

    def degrade(img):
        img = tf.cast(img, tf.float32)[..., None] / 255.0
        low = tf.image.resize(img, (low_size, low_size), method="area")
        blurred = tf.image.resize(low, (size, size), method="bicubic")
        return tf.clip_by_value(blurred, 0.0, 1.0), img

    ds = tf.data.Dataset.from_tensor_slices(clean)
    if shuffle:
        ds = ds.shuffle(len(clean), seed=seed)
    return ds.map(degrade, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)
