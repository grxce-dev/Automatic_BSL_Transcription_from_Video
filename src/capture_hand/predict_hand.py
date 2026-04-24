"""
predict_hand.py
---------------
Trains an LSTM model on recorded hand landmark sequences for BSL gesture classification.

Loads .npy sequence files from data/hand/<class_name>/, trains an LSTM classifier,
and saves the model, training history plot, and confusion matrix to models/evaluation/hand.

Output:
    models/detection_model/hand_model.h5
    models/evaluation/hand/training_history.png
    models/evaluation/hand/confusion_matrix.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from keras.models import Sequential
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


# Configuration
data_path = 'data/hand'         
sequence_length = 30        
num_features = 126

# Load data
def load_data():
    """
    Docstring for load_data
    """
    sequences = []           
    class_numbers = []                
    class_names = [] 

    for folder in os.listdir(data_path):
        folder_path = os.path.join(data_path, folder)

        if os.path.isdir(folder_path):
            class_names.append(folder)

    class_names.sort()
    label_map = {name: idx for idx, name in enumerate(class_names)}

    np.save("models/hand/hand_class_names.npy", class_names)
    
    # Load each sequence
    for class_name in class_names:
        class_path = os.path.join(data_path, class_name)

        for file in os.listdir(class_path):
            if file.endswith(".npy"):
                filepath = os.path.join(class_path, file)
                sequence = np.load(filepath)

                if sequence.shape == (sequence_length, num_features):
                    sequences.append(sequence)
                    class_numbers.append(label_map[class_name])
                else:
                    print("Skipped:", file, "Shape:", sequence.shape)

    return np.array(sequences), np.array(class_numbers), class_names

# Normalization
def normalize_data(input_data):
    """
    Docstring for normalize_data
    
    :param X: Description
    """
    mean = np.mean(input_data, axis = (1, 2), keepdims = True)
    std  = np.std(input_data, axis = (1, 2), keepdims = True) + 1e-8
    return (input_data - mean) / std

# Load dataset
sequences, class_numbers, class_names = load_data() 
 
if len(sequences) == 0:
    raise ValueError("No data found — check data path and that .npy files exist.")

sequences = normalize_data(sequences)
class_numbers = to_categorical(class_numbers, num_classes=len(class_names))

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(sequences, class_numbers, test_size = 0.2, random_state = 42) # Currently 80/20

# Build model
model = Sequential([
    LSTM(32, input_shape=(sequence_length, num_features)),
    Dropout(0.4),                                        # Prevent overfitting
    Dense(32, activation = "relu"),                      # Dense decision layer
    Dropout(0.3),
    Dense(len(class_names), activation = "softmax")      # Output layer (multi-class softmax)
])

model.compile(
    optimizer = "adam",
    loss = "categorical_crossentropy",
    metrics = ["accuracy"]
)

model.summary()

# Early stopping to prevent overfitting
early_stop = EarlyStopping(
    monitor = "val_loss",
    patience = 5,
    restore_best_weights = True
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


plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label = 'Train')
plt.plot(history.history['val_accuracy'], label = 'Validation')
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
plt.savefig("models/evaluation/hand/training_history.png")
plt.show()

cm = confusion_matrix(true_labels, pred_labels)
display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = class_names)
figure, ax = plt.subplots(figsize=(14, 14))
display.plot(ax=ax, xticks_rotation = 45)

plt.tight_layout()
plt.savefig("models/evaluation/hand/confusion_matrix.png")
os.makedirs("models/evaluation/hand", exist_ok = True)
plt.show()


# Save model
os.makedirs("models", exist_ok = True)
model.save(f"models/detection_model/hand_model.h5")
os.makedirs("models/detection_model", exist_ok = True)
print("Model saved to models")
