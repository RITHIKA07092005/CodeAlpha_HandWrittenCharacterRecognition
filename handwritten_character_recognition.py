# ============================================================
#  TASK 3: Handwritten Character Recognition — CodeAlpha
#  Dataset : EMNIST Letters (from Kaggle)
#  Model   : CNN (Convolutional Neural Network)
#  Platform: Google Colab
# ============================================================

# ── CELL 1 ── Install / Import everything needed
# Run this first — it installs the Kaggle API
!pip install kaggle --quiet

import os, zipfile, struct, numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix

print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))


# ── CELL 2 ── Upload your Kaggle API key
# 1. Go to https://www.kaggle.com → Your Account → API → "Create New Token"
# 2. A file called  kaggle.json  will download to your computer
# 3. Run this cell and upload that file when prompted

from google.colab import files
files.upload()          # upload kaggle.json here

# Move the key to the right place so the Kaggle CLI can find it
os.makedirs("/root/.kaggle", exist_ok=True)
!cp kaggle.json /root/.kaggle/kaggle.json
!chmod 600 /root/.kaggle/kaggle.json
print("Kaggle key set up ✓")


# ── CELL 3 ── Download the EMNIST dataset from Kaggle
# Dataset page: https://www.kaggle.com/datasets/crawford/emnist
!kaggle datasets download -d crawford/emnist --quiet
print("Download complete ✓")

# Unzip
with zipfile.ZipFile("emnist.zip", "r") as z:
    z.extractall("emnist_data")
print("Files extracted:")
!ls emnist_data/


# ── CELL 4 ── Load EMNIST Letters using the binary (.ubyte) files
# The EMNIST binary format is identical to the classic MNIST format

def load_emnist_binary(images_path, labels_path):
    """Read EMNIST binary files and return numpy arrays."""
    with open(images_path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num, rows, cols)

    with open(labels_path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)

    return images, labels

# ── Update these paths if the zip extracted into a sub-folder ──
BASE = "emnist_data/gzip"          # adjust if needed after listing files above

train_images, train_labels = load_emnist_binary(
    f"{BASE}/emnist-letters-train-images-idx3-ubyte",
    f"{BASE}/emnist-letters-train-labels-idx1-ubyte"
)
test_images, test_labels = load_emnist_binary(
    f"{BASE}/emnist-letters-test-images-idx3-ubyte",
    f"{BASE}/emnist-letters-test-labels-idx1-ubyte"
)

print(f"Train images : {train_images.shape}  |  Labels : {train_labels.shape}")
print(f"Test  images : {test_images.shape}   |  Labels : {test_labels.shape}")
print(f"Unique labels: {np.unique(train_labels)}")   # 1-26  (a-z)


# ── CELL 5 ── Pre-process the data

# EMNIST images are transposed by design — fix that
train_images = np.transpose(train_images, (0, 2, 1))
test_images  = np.transpose(test_images,  (0, 2, 1))

# Normalize pixel values to [0, 1]
train_images = train_images.astype("float32") / 255.0
test_images  = test_images.astype("float32")  / 255.0

# Add channel dimension  →  (N, 28, 28, 1)
train_images = train_images[..., np.newaxis]
test_images  = test_images[..., np.newaxis]

# EMNIST Letters labels are 1-26; shift to 0-25 for one-hot encoding
train_labels = train_labels - 1
test_labels  = test_labels  - 1
NUM_CLASSES  = 26

# One-hot encode
train_labels_oh = keras.utils.to_categorical(train_labels, NUM_CLASSES)
test_labels_oh  = keras.utils.to_categorical(test_labels,  NUM_CLASSES)

print("Pre-processing done ✓")
print(f"train_images shape: {train_images.shape}")
print(f"train_labels_oh shape: {train_labels_oh.shape}")


# ── CELL 6 ── Visualize some samples
ALPHABET = [chr(i) for i in range(ord('A'), ord('Z')+1)]   # A-Z

fig, axes = plt.subplots(4, 8, figsize=(14, 7))
for i, ax in enumerate(axes.flat):
    ax.imshow(train_images[i].squeeze(), cmap="gray")
    ax.set_title(ALPHABET[train_labels[i]], fontsize=10)
    ax.axis("off")
plt.suptitle("Sample EMNIST Letters", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("samples.png", dpi=100)
plt.show()
print("Sample grid saved as samples.png")


# ── CELL 7 ── Build the CNN model

def build_cnn(num_classes=26):
    model = keras.Sequential([
        # ── Block 1 ──
        layers.Conv2D(32, (3,3), padding="same", activation="relu",
                      input_shape=(28, 28, 1)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3,3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        # ── Block 2 ──
        layers.Conv2D(64, (3,3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3,3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        # ── Block 3 ──
        layers.Conv2D(128, (3,3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        # ── Classifier ──
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax")
    ], name="EMNIST_CNN")
    return model

model = build_cnn(NUM_CLASSES)
model.summary()


# ── CELL 8 ── Compile the model

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Callbacks: reduce LR on plateau + early stopping
callbacks = [
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                      patience=3, verbose=1, min_lr=1e-6),
    keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=8,
                                  restore_best_weights=True, verbose=1),
    keras.callbacks.ModelCheckpoint("best_model.keras",
                                    save_best_only=True, monitor="val_accuracy",
                                    verbose=1)
]
print("Model compiled ✓")


# ── CELL 9 ── Train the model  (~10-15 min on Colab GPU)

BATCH_SIZE = 128
EPOCHS     = 30

history = model.fit(
    train_images, train_labels_oh,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.1,      # 10 % of train used for validation
    callbacks=callbacks,
    verbose=1
)
print("\nTraining complete ✓")


# ── CELL 10 ── Plot training curves

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(history.history["accuracy"],     label="Train Acc")
ax1.plot(history.history["val_accuracy"], label="Val Acc")
ax1.set_title("Accuracy over Epochs"); ax1.legend(); ax1.grid(True)

ax2.plot(history.history["loss"],     label="Train Loss")
ax2.plot(history.history["val_loss"], label="Val Loss")
ax2.set_title("Loss over Epochs"); ax2.legend(); ax2.grid(True)

plt.tight_layout()
plt.savefig("training_curves.png", dpi=100)
plt.show()
print("Training curves saved ✓")


# ── CELL 11 ── Evaluate on the test set

test_loss, test_acc = model.evaluate(test_images, test_labels_oh, verbose=0)
print(f"\n📊 Test Accuracy : {test_acc*100:.2f} %")
print(f"   Test Loss     : {test_loss:.4f}")


# ── CELL 12 ── Full classification report

y_pred_probs = model.predict(test_images, verbose=0)
y_pred       = np.argmax(y_pred_probs, axis=1)

print("\n=== Classification Report ===")
print(classification_report(test_labels, y_pred,
                             target_names=ALPHABET))


# ── CELL 13 ── Confusion matrix (heatmap)

cm = confusion_matrix(test_labels, y_pred)

plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=ALPHABET, yticklabels=ALPHABET,
            linewidths=0.5)
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label",      fontsize=12)
plt.title("Confusion Matrix — EMNIST Letters", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=100)
plt.show()
print("Confusion matrix saved ✓")


# ── CELL 14 ── Predict and visualize on individual test samples

def predict_single(index):
    img   = test_images[index]
    true  = ALPHABET[test_labels[index]]
    probs = model.predict(img[np.newaxis, ...], verbose=0)[0]
    pred  = ALPHABET[np.argmax(probs)]
    conf  = np.max(probs) * 100

    plt.figure(figsize=(3, 3))
    plt.imshow(img.squeeze(), cmap="gray")
    color = "green" if pred == true else "red"
    plt.title(f"True: {true}  |  Pred: {pred}  ({conf:.1f}%)", color=color)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    return pred, true, conf

# Show 6 random predictions
for idx in np.random.choice(len(test_images), 6, replace=False):
    predict_single(idx)


# ── CELL 15 ── Save the final model + download it

model.save("emnist_cnn_final.keras")
print("Model saved as emnist_cnn_final.keras ✓")

# Download all output files to your computer
from google.colab import files
for fname in ["emnist_cnn_final.keras", "best_model.keras",
              "training_curves.png", "confusion_matrix.png", "samples.png"]:
    try:
        files.download(fname)
        print(f"Downloading {fname} ✓")
    except Exception as e:
        print(f"Could not download {fname}: {e}")

print("\n✅ All done! Task 3 complete.")
