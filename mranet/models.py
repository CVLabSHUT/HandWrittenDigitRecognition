"""Model definitions: MRA module, MRA U-Net, language and digit classifiers, baselines."""
import keras
from keras import layers


def mra_module(x, filters=64, name="mra"):
    """Multi-resolution attention module.

    Three parallel convolutions (1x1, 3x3, 5x5) are summed and projected with a
    1x1 convolution. A channel attention vector is computed from the globally
    pooled features and multiplies the projected feature map.
    """
    a = layers.Conv2D(filters, 1, padding="same", activation="relu", name=f"{name}_conv1x1")(x)
    b = layers.Conv2D(filters, 3, padding="same", activation="relu", name=f"{name}_conv3x3")(x)
    c = layers.Conv2D(filters, 5, padding="same", activation="relu", name=f"{name}_conv5x5")(x)
    merged = layers.Add(name=f"{name}_sum")([a, b, c])

    features = layers.Conv2D(filters, 1, padding="same", activation="relu", name=f"{name}_proj")(merged)
    attention = layers.GlobalAveragePooling2D(keepdims=True, name=f"{name}_gap")(features)
    attention = layers.Conv2D(filters, 1, activation="relu", name=f"{name}_att1")(attention)
    attention = layers.BatchNormalization(name=f"{name}_att_bn")(attention)
    attention = layers.Conv2D(filters, 1, activation="relu", name=f"{name}_att2")(attention)
    return layers.Multiply(name=f"{name}_out")([features, attention])


def _double_conv(x, filters, name):
    x = layers.Conv2D(filters, 3, padding="same", activation="relu", name=f"{name}_conv1")(x)
    x = layers.Conv2D(filters, 3, padding="same", activation="relu", name=f"{name}_conv2")(x)
    return layers.BatchNormalization(name=f"{name}_bn")(x)


def build_mra_unet(input_shape=(128, 128, 1), base_filters=64, dropout=0.5):
    """U-Net with an MRA module on every skip connection.

    The input side length has to be divisible by 16 (four pooling steps).
    """
    inputs = keras.Input(shape=input_shape, name="image")
    x = inputs

    skips = []
    for level in range(4):
        features = _double_conv(x, base_filters * 2 ** level, f"enc{level + 1}")
        skips.append(mra_module(features, name=f"skip{level + 1}"))
        x = layers.MaxPooling2D(2, name=f"enc{level + 1}_pool")(features)
        x = layers.Dropout(dropout, name=f"enc{level + 1}_drop")(x)

    x = _double_conv(x, base_filters * 16, "bottleneck1")
    x = _double_conv(x, base_filters * 16, "bottleneck2")

    for level in reversed(range(4)):
        filters = base_filters * 2 ** level
        x = layers.Conv2DTranspose(filters, 3, strides=2, padding="same", name=f"dec{level + 1}_up")(x)
        x = layers.Concatenate(name=f"dec{level + 1}_concat")([x, skips[level]])
        x = layers.Dropout(dropout, name=f"dec{level + 1}_drop")(x)
        x = _double_conv(x, filters, f"dec{level + 1}")

    outputs = layers.Conv2D(1, 1, activation="sigmoid", name="restored")(x)
    return keras.Model(inputs, outputs, name="mra_unet")


def build_language_classifier(input_shape=(128, 128, 1), n_languages=12):
    """CNN with an MRA module after every pooling step; predicts the script of a digit image."""
    inputs = keras.Input(shape=input_shape, name="image")
    x = inputs
    for block, filters in enumerate((64, 128, 256), start=1):
        x = layers.Conv2D(filters, 3, padding="same", activation="relu", name=f"conv{block}a")(x)
        x = layers.Conv2D(filters, 3, padding="same", activation="relu", name=f"conv{block}b")(x)
        x = layers.MaxPooling2D(2, name=f"pool{block}")(x)
        x = mra_module(x, name=f"mra{block}")
    x = layers.Flatten(name="flatten")(x)
    for i, units in enumerate((512, 256, 128), start=1):
        x = layers.Dense(units, activation="relu", name=f"dense{i}")(x)
        x = layers.Dropout(0.25, name=f"dense{i}_drop")(x)
    outputs = layers.Dense(n_languages, activation="softmax", name="language")(x)
    return keras.Model(inputs, outputs, name="language_classifier")


def build_digit_classifier(language_model, n_classes=10):
    """Digit classifier for one language.

    The convolutional part of the language classifier (everything up to the
    flatten layer) is reused and frozen; only the new dense head is trained.
    """
    features = language_model.get_layer("flatten").output
    backbone = keras.Model(language_model.input, features, name="frozen_backbone")
    backbone.trainable = False

    x = layers.Dense(256, activation="relu", name="head_dense1")(backbone.output)
    x = layers.Dropout(0.25, name="head_drop1")(x)
    x = layers.Dense(128, activation="relu", name="head_dense2")(x)
    x = layers.Dropout(0.25, name="head_drop2")(x)
    outputs = layers.Dense(n_classes, activation="softmax", name="digit")(x)
    return keras.Model(backbone.input, outputs, name="digit_classifier")


def with_frozen_unet(unet, classifier, name):
    """Chain a frozen restoration U-Net in front of a classifier."""
    unet.trainable = False
    inputs = keras.Input(shape=unet.input_shape[1:], name="image")
    restored = unet(inputs, training=False)
    return keras.Model(inputs, classifier(restored), name=name)


# ---- baselines used for comparison ------------------------------------------

def build_lenet5(input_shape=(32, 32, 1), n_classes=10):
    return keras.Sequential([
        keras.Input(shape=input_shape),
        layers.Conv2D(6, 5, padding="same", activation="tanh"),
        layers.AveragePooling2D(2),
        layers.Conv2D(16, 5, activation="tanh"),
        layers.AveragePooling2D(2),
        layers.Flatten(),
        layers.Dense(120, activation="tanh"),
        layers.Dense(84, activation="tanh"),
        layers.Dense(n_classes, activation="softmax"),
    ], name="lenet5")


def build_vgg16(input_shape=(32, 32, 1), n_classes=10, weights="imagenet"):
    """Frozen VGG16 feature extractor with a small dense head; grayscale input is repeated to 3 channels."""
    height, width, _ = input_shape
    base = keras.applications.VGG16(weights=weights, include_top=False, input_shape=(height, width, 3))
    base.trainable = False
    inputs = keras.Input(shape=input_shape)
    x = layers.Concatenate()([inputs, inputs, inputs])
    x = base(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="vgg16")
