
# --------------------------------------------------------------------------- #
# Imports
# --------------------------------------------------------------------------- #
import os
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, ConfusionMatrixDisplay)

from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau

# Base models
from tensorflow.keras.applications import (
    MobileNet, EfficientNetV2L, VGG16, DenseNet201, EfficientNetB7
)
from keras.applications.convnext import ConvNeXtBase

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
INPUT_SHAPE = (128, 128, 3)
LEARNING_RATE = 1e-4
EPOCHS = 20
BATCH_SIZE = 32
MODEL_DIR = Path("models").resolve()
MODEL_DIR.mkdir(exist_ok=True)

# --------------------------------------------------------------------------- #
# Data (expects X_train, X_test, y_train, y_test to be pre-loaded np.arrays)
# --------------------------------------------------------------------------- #
# TODO: replace with your own data-loading pipeline
# X_train, X_test, y_train, y_test = ...

# --------------------------------------------------------------------------- #
# Callbacks
# --------------------------------------------------------------------------- #
callbacks = [
    EarlyStopping(patience=4, restore_best_weights=True),
    ReduceLROnPlateau(patience=2, factor=0.5, verbose=1)
]

# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #
def build_model(base_model, name: str) -> Model:
    """Freeze base layers and add a simple classifier head."""
    base_model.trainable = False
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=base_model.input, outputs=output, name=name)
    model.compile(optimizer=Adam(LEARNING_RATE),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model


def train_and_evaluate(model: Model,
                       model_name: str,
                       data: Tuple[np.ndarray, ...]) -> Dict[str, float]:
    """Train model, evaluate on test set, save artefacts, return metrics."""
    X_train, X_test, y_train, y_test = data
    print(f"\n—— Training {model_name} ——")
    model.fit(X_train, y_train,
              validation_data=(X_test, y_test),
              epochs=EPOCHS,
              batch_size=BATCH_SIZE,
              callbacks=callbacks,
              verbose=2)

    # Metrics
    y_prob = model.predict(X_test, verbose=0)
    y_pred = (y_prob > 0.5).astype(int)
    report = classification_report(
        y_test, y_pred,
        target_names=["Benign", "Malignant"],
        output_dict=True
    )
    accuracy = report["accuracy"]
    f1 = report["Malignant"]["f1-score"]
    sensitivity = report["Malignant"]["recall"]
    roc_auc = roc_auc_score(y_test, y_prob)

    print(f"{model_name:18s}  Acc: {accuracy:.3f} | "
          f"F1: {f1:.3f} | Sensitivity: {sensitivity:.3f} | AUC: {roc_auc:.3f}")

    # Confusion-matrix plot
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["Benign", "Malignant"]).plot(
        cmap="Blues", values_format="d")
    plt.title(model_name)
    plt.show()

    # Save model
    model_path = MODEL_DIR / f"{model_name}.keras"
    model.save(model_path)
    return {
        "name": model_name,
        "accuracy": accuracy,
        "f1": f1,
        "sensitivity": sensitivity,
        "auc": roc_auc
    }


def plot_roc_curves(curves: Dict[str, Tuple[np.ndarray, ...]]) -> None:
    """Plot ROC curves for all models in one figure."""
    plt.figure(figsize=(8, 6))
    for label, (fpr, tpr, auc_val) in curves.items():
        plt.plot(fpr, tpr, label=f"{label} (AUC {auc_val:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
    plt.title("ROC Curves — All Models")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.show()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Prepare data tuple for convenience
    data_tuple = (X_train, X_test, y_train, y_test)

    # Dictionary of base architectures
    base_models = {
        "MobileNet":       MobileNet(weights="imagenet", include_top=False,
                                     input_shape=INPUT_SHAPE),
        "EfficientNetV2L": EfficientNetV2L(weights="imagenet", include_top=False,
                                           input_shape=INPUT_SHAPE),
        "VGG16":           VGG16(weights="imagenet", include_top=False,
                                 input_shape=INPUT_SHAPE),
        "DenseNet201":     DenseNet201(weights="imagenet", include_top=False,
                                       input_shape=INPUT_SHAPE),
        "EfficientNetB7":  EfficientNetB7(weights="imagenet", include_top=False,
                                          input_shape=INPUT_SHAPE),
        "ConvNeXtBase":    ConvNeXtBase(weights="imagenet", include_top=False,
                                        input_shape=INPUT_SHAPE)
        # Add GhostNet V1/V2/V3 here when ready
    }

    metrics, roc_curves = {}, {}

    # Train loop
    for name, backbone in base_models.items():
        model = build_model(backbone, name)
        result = train_and_evaluate(model, name, data_tuple)

        # Store metrics & ROC
        metrics[name] = result
        fpr, tpr, _ = roc_curve(y_test, model.predict(X_test))
        roc_curves[name] = (fpr, tpr, result["auc"])

    # Plot aggregate ROC
    plot_roc_curves(roc_curves)

    # Bar-chart comparison
    plt.figure(figsize=(8, 5))
    plt.bar(metrics.keys(),
            [m["accuracy"] for m in metrics.values()],
            color="steelblue")
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy Comparison")
    plt.ylim(0, 1)
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.show()
