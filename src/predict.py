# Responsible for: Load trained model, Run real-time inference, Display prediction

import os
import numpy as np
from sklearn.model_selection import train_test_split

from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

DATA_PATH = "data"          
SEQUENCE_LENGTH = 35        
NUM_FEATURES = 126          # 2 hands × 21 landmarks × (x,y,z)

# LOAD DATA
def load_data():
    X = []                  # store sequences
    y = []                  # store numeric labels
    class_names = []        # List of sign names

    # Every top-level folder = one class
    for folder in os.listdir(DATA_PATH):
        folder_path = os.path.join(DATA_PATH, folder)

        if os.path.isdir(folder_path):
            class_names.append(folder)

    # Sort for consistent ordering
    class_names.sort()

    # Map class name to an integer label
    label_map = {name: idx for idx, name in enumerate(class_names)}

    # TESTING
    print("Class names:", class_names)
    print("Label map:", label_map)

    # Now load each sequence
    for class_name in class_names:
        class_path = os.path.join(DATA_PATH, class_name)

        for variation in os.listdir(class_path):
            variation_path = os.path.join(class_path, variation)

            if not os.path.isdir(variation_path):
                continue

            for file in os.listdir(variation_path):
                if file.endswith(".npy"):
                    filepath = os.path.join(variation_path, file)
                    sequence = np.load(filepath)

                    # Ensure correct shape
                    if sequence.shape == (SEQUENCE_LENGTH, NUM_FEATURES):
                        X.append(sequence)
                        y.append(label_map[class_name])
                    else:
                        print("Skipped:", file, "Shape:", sequence.shape)

    return np.array(X), np.array(y), class_names

# Load dataset
X, y, class_names = load_data()

print("Detected classes:", class_names)
print("Dataset shape:", X.shape)   # Should be (samples, 35, 126)

# NORMALISATION 
def normalize_data(X):
    """
    Standardise each sequence independently.
    Reduces scale and position bias.
    """
    mean = np.mean(X, axis=(1, 2), keepdims=True)
    std = np.std(X, axis=(1, 2), keepdims=True) + 1e-8
    return (X - mean) / std

X = normalize_data(X)

# HOT ENCODE LABELS (Multi-class format)
y = to_categorical(y, num_classes=len(class_names))

# TRAIN / TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training samples:", X_train.shape[0])
print("Test samples:", X_test.shape[0])

# BUILD LSTM MODEL
model = Sequential([
    LSTM(32, input_shape=(SEQUENCE_LENGTH, NUM_FEATURES)),
    Dropout(0.4),                                      # Prevent overfitting
    Dense(32, activation="relu"),                      # Dense decision layer
    Dropout(0.3),
    Dense(len(class_names), activation="softmax")      # Output layer (multi-class softmax)
])


# COMPILE MODEL
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",   # multi-class
    metrics=["accuracy"]
)

model.summary()

# EARLY STOPPING (Prevents Overfitting)
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)


# TRAIN MODEL
history = model.fit(
    X_train,
    y_train,
    epochs=40,
    batch_size=16,
    validation_data=(X_test, y_test),
    callbacks=[early_stop]
)

# EVALUATE MODEL
loss, accuracy = model.evaluate(X_test, y_test)
print("Final Test Accuracy:", accuracy)

# SAVE MODEL
SAVE_FOLDER = "models/"
os.makedirs(SAVE_FOLDER, exist_ok = True)
file_count = len(os.listdir(SAVE_FOLDER))
model.save(f"models/bsl_multiclass_model_{file_count}.h5")
print("Model saved")
