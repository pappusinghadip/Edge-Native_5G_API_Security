"""1D-CNN architecture definition for the centralized and FL models."""

from __future__ import annotations

import platform
from typing import Any

MODEL_SPEC = {
    "input_shape": (10, 1),
    "num_classes": 2,
    "conv_filters": (64, 128, 128),
    "kernel_size": 3,
    "dense_units": (256, 128),
    "dropout_rates": (0.4, 0.3),
    "optimizer": "adam",
    "learning_rate": 0.001,
    "momentum": 0.9,
    "use_batch_norm": True,
    "use_focal_loss": True,
    "focal_alpha": 0.75,
    "focal_gamma": 2.0,
    "label_smoothing": 0.1,
}


def focal_loss(alpha: float = 0.75, gamma: float = 2.0, class_weighted: bool = True) -> Any:
    """Create a focal loss function for imbalanced classification.

    Focal loss down-weights well-classified examples and focuses training
    on hard, misclassified examples.
    """
    import tensorflow as tf

    def _focal_loss(y_true: Any, y_pred: Any) -> Any:
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())
        if class_weighted:
            # alpha on the malicious class (column 1), 1 - alpha on benign (column 0)
            alpha_weights = y_true * tf.constant([1.0 - alpha, alpha], dtype=y_pred.dtype)
        else:
            # Pre-round-5 behaviour, kept only to reproduce earlier runs. With one-hot targets
            # the true-class column always gets alpha, so every example is weighted alike.
            alpha_weights = y_true * alpha + (1.0 - y_true) * (1.0 - alpha)
        focal_weights = alpha_weights * tf.pow(1.0 - y_pred, gamma)
        cross_entropy = -y_true * tf.math.log(y_pred)
        loss = focal_weights * cross_entropy
        return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))

    _focal_loss.__name__ = "focal_loss"
    return _focal_loss


def build_model(
    input_shape: tuple[int, int] = (10, 1),
    num_classes: int = 2,
    conv_filters: tuple[int, ...] = (64, 128, 128),
    kernel_size: int = 3,
    dense_units: tuple[int, ...] = (256, 128),
    dropout_rates: tuple[float, ...] = (0.4, 0.3),
    learning_rate: float = 0.001,
    momentum: float = 0.9,
    optimizer: str = "adam",
    use_batch_norm: bool = True,
    use_focal_loss: bool = True,
    focal_alpha: float = 0.75,
    focal_gamma: float = 2.0,
    label_smoothing: float = 0.1,
    focal_class_weighted: bool = True,
) -> Any:
    """Build and compile the thesis 1D-CNN with mixed pooling.

    Architecture:
    - Input (10, 1): features as time steps so Conv1D kernels slide across them.
    - 3 Conv1D blocks with BatchNorm for deeper feature extraction.
    - Mixed pooling (GlobalMax + GlobalAvg concatenated) captures both peak
      activations and average response — critical for network traffic where
      both extreme values and averages are discriminative.
    - Dense head with dropout for classification.
    - Focal loss to handle severe class imbalance.
    """
    try:
        import tensorflow as tf
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("TensorFlow is required to build the CNN model.") from exc

    inp = tf.keras.layers.Input(shape=input_shape)
    x = inp

    # Convolutional feature extraction
    for filters in conv_filters:
        x = tf.keras.layers.Conv1D(filters, kernel_size, activation="relu", padding="same")(x)
        if use_batch_norm:
            x = tf.keras.layers.BatchNormalization()(x)

    # Mixed pooling: captures both peak and average patterns
    max_pool = tf.keras.layers.GlobalMaxPooling1D()(x)
    avg_pool = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Concatenate()([max_pool, avg_pool])

    # Dense classification head
    for units, drop_rate in zip(dense_units, dropout_rates, strict=True):
        x = tf.keras.layers.Dense(units, activation="relu")(x)
        x = tf.keras.layers.Dropout(drop_rate)(x)

    output = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=inp, outputs=output)
    opt = build_optimizer(tf, optimizer=optimizer, learning_rate=learning_rate, momentum=momentum)

    if use_focal_loss:
        loss_fn = focal_loss(alpha=focal_alpha, gamma=focal_gamma, class_weighted=focal_class_weighted)
    else:
        loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)

    model.compile(
        optimizer=opt,
        loss=loss_fn,
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def build_optimizer(tf: Any, optimizer: str = "adam", learning_rate: float = 0.001, momentum: float = 0.9) -> Any:
    """Select an optimizer with Apple Silicon compatibility when needed."""
    legacy_optimizers = getattr(tf.keras.optimizers, "legacy", None)
    use_legacy = legacy_optimizers is not None and platform.system() == "Darwin" and platform.machine() == "arm64"

    if optimizer == "adam":
        if use_legacy:
            return legacy_optimizers.Adam(learning_rate=learning_rate)
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)

    if use_legacy:
        return legacy_optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
    return tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)


def build_model_from_config(model_config: dict[str, Any]) -> Any:
    """Build a compiled model from the YAML model configuration."""
    cfg = model_config["model"]
    return build_model(
        input_shape=tuple(cfg["input_shape"]),
        num_classes=int(cfg["num_classes"]),
        conv_filters=tuple(cfg["conv_filters"]),
        kernel_size=int(cfg["kernel_size"]),
        dense_units=tuple(cfg["dense_units"]),
        dropout_rates=tuple(cfg["dropout_rates"]),
        learning_rate=float(cfg["learning_rate"]),
        momentum=float(cfg.get("momentum", 0.9)),
        optimizer=str(cfg.get("optimizer", "adam")),
        use_batch_norm=bool(cfg.get("use_batch_norm", True)),
        use_focal_loss=bool(cfg.get("use_focal_loss", True)),
        focal_alpha=float(cfg.get("focal_alpha", 0.75)),
        focal_gamma=float(cfg.get("focal_gamma", 2.0)),
        focal_class_weighted=bool(cfg.get("focal_class_weighted", True)),
        label_smoothing=float(cfg.get("label_smoothing", 0.1)),
    )


def model_summary_to_dict(model: Any) -> dict[str, Any]:
    """Return a minimal summary representation for logging."""
    return {
        "layers": [
            {
                "name": layer.name,
                "type": layer.__class__.__name__,
                "params": int(layer.count_params()),
            }
            for layer in model.layers
        ],
        "total_params": int(model.count_params()),
        "trainable_params": int(sum(layer.count_params() for layer in model.layers if layer.trainable)),
    }
