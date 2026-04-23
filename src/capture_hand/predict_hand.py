# Predict Hand Landmarks

import os
import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

# Configuration
data_path= 'data/hand'         
sequence_length = 30        
num_features = 126          # 2 hands × 21 landmarks × (x,y,z)

# Load data
def load_data():
    X = []                  # store sequences
    y = []                  # store numeric labels
    class_names = []        # List of sign names

    # Every top-level folder = one class
    for folder in os.listdir(data_path):
        folder_path = os.path.join(data_path, folder)

        if os.path.isdir(folder_path):
            class_names.append(folder)

    class_names.sort()
    label_map = {name: idx for idx, name in enumerate(class_names)}

    np.save("models/hand_class_names.npy", class_names)
    
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
print("Dataset shape:",    X.shape)   
 
if len(X) == 0:
    raise ValueError("No data found — check data path and that .npy files exist.")
 
#print("Detected classes:", class_names)
#print("Dataset shape:", X.shape) 

# Normalisation
def normalize_data(X):
    mean = np.mean(X, axis = (1, 2), keepdims = True)
    std  = np.std(X, axis = (1, 2), keepdims = True) + 1e-8
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

# Early stopping to prevent overfitting
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# Train model
history = model.fit(
    X_train, y_train,
    epochs = 30,
    batch_size = 16,
    validation_data = (X_test, y_test),
    callbacks = [early_stop],
    shuffle = True
)

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print("Final Test Accuracy:", accuracy)

preds = model.predict(X_test)
pred_labels = np.argmax(preds, axis=1)
true_labels = np.argmax(y_test, axis=1)

for i in range(10):
    print("Pred:", class_names[pred_labels[i]],
          "True:", class_names[true_labels[i]])

# Save model
os.makedirs("models", exist_ok = True)
file_count = len(os.listdir("models"))
model.save(f"models/detection_model/model_acc_{accuracy:.2f}.h5")
print("Model saved to models")
