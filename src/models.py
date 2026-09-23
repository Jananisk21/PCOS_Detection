"""
Ensemble transfer-learning model: EfficientNetB7 + DenseNet201.

This is the "EffiDenseGenOp" architecture highlighted as the best-performing
approach in the base paper (Sections III-D and IV / base study row of
Table 1): two ImageNet-pretrained backbones are used as parallel feature
extractors, their pooled features are concatenated, and a small tunable
classifier head (dense width, dropout, learning rate, optimizer -- the
genes optimized by the Genetic Algorithm in genetic_optimizer.py) sits on
top.

Note on pretrained weights: this sandbox environment has no internet
access to download the ImageNet weight files, so `weights="imagenet"`
will automatically and transparently fall back to random initialization
here with a printed warning (`build_ensemble(..., try_pretrained=True)`).
When you run this project on a machine with normal internet access,
Keras will download the real ImageNet weights on first use, exactly as
the base paper intends -- no code changes needed.
"""

import tensorflow as tf
from tensorflow.keras import layers, models


def _build_backbone(name: str, input_shape, try_pretrained: bool):
    weights = "imagenet" if try_pretrained else None
    common_kwargs = dict(include_top=False, input_shape=input_shape, pooling="avg")

    def _instantiate(w):
        if name == "efficientnetb7":
            return tf.keras.applications.EfficientNetB7(weights=w, **common_kwargs)
        elif name == "efficientnetb0":
            return tf.keras.applications.EfficientNetB0(weights=w, **common_kwargs)
        elif name == "densenet201":
            return tf.keras.applications.DenseNet201(weights=w, **common_kwargs)
        elif name == "densenet121":
            return tf.keras.applications.DenseNet121(weights=w, **common_kwargs)
        else:
            raise ValueError(f"Unknown backbone: {name}")

    if try_pretrained:
        try:
            backbone = _instantiate("imagenet")
            return backbone
        except Exception as e:
            print(f"[models.py] Could not download ImageNet weights for {name} "
                  f"({e}). Falling back to random initialization. On a machine "
                  f"with normal internet access this will use real pretrained weights.")
    return _instantiate(None)


def build_feature_extractor(input_shape=(128, 128, 3), backbone_variant="full",
                             try_pretrained=True):
    """
    Builds the frozen dual-backbone feature extractor only (no classifier
    head). Because the backbones are always frozen in this project (standard
    practice for transfer learning on a small medical-imaging dataset), their
    output features can be computed ONCE per image and cached/reused across
    every Genetic-Algorithm candidate and every training epoch, instead of
    re-running two large CNNs on every forward pass. This is purely a
    performance optimization -- it does not change the model architecture
    or the EffiDenseGenOp methodology (frozen pretrained CNN + trainable
    dense head is exactly what "transfer learning" means here and in the
    base paper).
    """
    inputs = layers.Input(shape=input_shape, name="input_image")

    if backbone_variant == "full":
        eff_name, dense_name = "efficientnetb7", "densenet201"
    elif backbone_variant == "lite":
        eff_name, dense_name = "efficientnetb0", "densenet121"
    else:
        raise ValueError(f"Unknown backbone_variant: {backbone_variant}")

    # NOTE: Keras functional Model names (e.g. "efficientnetb7", "densenet201")
    # cannot be reassigned after construction, so these sub-models keep their
    # default keras.applications names when nested below. gradcam.py locates
    # the EfficientNet backbone by name keyword rather than a fixed name.
    eff_backbone = _build_backbone(eff_name, input_shape, try_pretrained)
    dense_backbone = _build_backbone(dense_name, input_shape, try_pretrained)
    eff_backbone.trainable = False
    dense_backbone.trainable = False

    eff_features = eff_backbone(tf.keras.applications.efficientnet.preprocess_input(inputs))
    dense_features = dense_backbone(tf.keras.applications.densenet.preprocess_input(inputs))
    fused = layers.Concatenate(name="ensemble_feature_fusion")([eff_features, dense_features])

    extractor = models.Model(inputs=inputs, outputs=fused, name="EffiDenseGenOp_feature_extractor")
    return extractor, eff_backbone, dense_backbone


def build_classifier_head(feature_dim, dense_units=256, dropout_rate=0.4,
                           learning_rate=1e-4, optimizer_name="adam"):
    """The small trainable classifier head that sits on top of the cached
    fused features. These are exactly the hyperparameters the Genetic
    Algorithm searches over."""
    inputs = layers.Input(shape=(feature_dim,), name="fused_features")
    x = layers.Dense(dense_units, activation="relu")(inputs)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(max(dense_units // 2, 8), activation="relu")(x)
    x = layers.Dropout(dropout_rate / 2)(x)
    output = layers.Dense(1, activation="sigmoid", name="pcos_probability")(x)

    head = models.Model(inputs=inputs, outputs=output, name="EffiDenseGenOp_head")
    optimizer = _make_optimizer(optimizer_name, learning_rate)
    head.compile(optimizer=optimizer, loss="binary_crossentropy",
                 metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
    return head


def combine_extractor_and_head(feature_extractor, head):
    """Chains the frozen feature extractor and the trained head into a
    single end-to-end model that takes raw images as input, for saving,
    prediction, and Grad-CAM use."""
    inputs = feature_extractor.input
    features = feature_extractor.output
    output = head(features)
    combined = models.Model(inputs=inputs, outputs=output, name="EffiDenseGenOp")
    combined.compile(optimizer=head.optimizer, loss="binary_crossentropy",
                      metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
    return combined


def _make_optimizer(name: str, lr: float):
    name = name.lower()
    if name == "adam":
        return tf.keras.optimizers.Adam(learning_rate=lr)
    elif name == "rmsprop":
        return tf.keras.optimizers.RMSprop(learning_rate=lr)
    elif name == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {name}")
