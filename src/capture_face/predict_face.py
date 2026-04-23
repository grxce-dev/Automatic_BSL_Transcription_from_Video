# Predict Face Landmarks

import os
import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt


# Configuration
data_path = 'data/face'
sequence_length = 10
num_features = 2

# Load Data
def load_data():
    X = []
    y = []
    class_names = []

    for folder in os.listdir(data_path):
        folder_path = os.path.join(data_path, folder)

        if os.path.isdir(folder_path):
            class_names.append(folder)

    class_names.sort()
    label_map = {name: index for index, name in enumerate(class_names)}

    np.save("models/face_class_names.npy", class_names)

    # Load each sequence
    for class_name in class_names:
        class_path = os.path.join(data_path, class_name)

        for file in os.listdir(class_path):
            if file.endswith(".npy"):
                filepath = os.path.join(class_path, file)
                sequence = np.load(filepath)

                if sequence.shape == (sequence_length, num_features):
                    X.append(sequence)
                    y.append(label_map[class_name])
                else:
                    print("Skipped:", file, "Shape:", sequence.shape)

    return np.array(X), np.array(y), class_names

# Load dataset
X, y, class_names = load_data()
 
print("Detected classes:", class_names)
print("Dataset shape:",    X.shape)   # (samples, 10, 2)
 
if len(X) == 0:
    raise ValueError("No data found — check data path and that .npy files exist.")

#print("Detected classes:", class_names)
#print("Dataset shape:", X.shape) 

# Normalisation
def normalize_data(X):
    mean = np.mean(X, axis = (1, 2), keepdims = True)
    std = np.std(X, axis = (1, 2), keepdims = True) + 1e-8
    return (X - mean) / std

X = normalize_data(X)

# Hot encode class names
y = to_categorical(y, num_classes=len(class_names))

# Train/Test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42) # Currently 80/20

# Build model
model = Sequential([
    LSTM(32, input_shape=(sequence_length, num_features)),
    Dropout(0.4),                                        # Prevent overfitting
    Dense(32, activation = "relu"),                      # Dense decision layer
    Dropout(0.3),
    Dense(len(class_names), activation = "softmax")      # Output layer (multi-class softmax)
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Early Stopping to prevent overfitting
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# Train Model
history = model.fit(
    X_train, y_train,
    epochs = 30,
    batch_size = 16,
    validation_data = (X_test, y_test),
    callbacks = [early_stop],
    shuffle = True
)

# Evaluate Model
loss, accuracy = model.evaluate(X_test, y_test)
print("Final Test Accuracy:", accuracy)

preds = model.predict(X_test)
pred_labels = np.argmax(preds, axis=1)
true_labels = np.argmax(y_test, axis=1)

for i in range(10):
    print("Pred:", class_names[pred_labels[i]],
          "True:", class_names[true_labels[i]])

# "PLOTTING THE ACTUAL AND PREDICTED VALUES, YOU CAN GAIN INSIGHTS INTO HOW WELL THE MODEL CATUPES THE UNDERLYING PATTERNS IN DATA"

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig("models/training_history.png")
plt.show()

os.makedirs("models", exist_ok = True)
model.save(f"models/detection_model/hand_model.h5")
print("Model saved to models")

print("Model saved to models")
