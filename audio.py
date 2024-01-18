import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import librosa

# Load data from an MP3 file using Librosa
mp3_file_path = r'E:\audio.detection\compressed\audio.mp3'
y, sr = librosa.load(mp3_file_path)

# Extract features (you may need to adjust these based on your requirements)
# Example: Mel-frequency cepstral coefficients (MFCCs)
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

# Assuming you have labels for your data
# Replace the following lines with your actual labels
labels = np.random.randint(2, size=len(mfccs[0]))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(mfccs.T, labels, test_size=0.2, random_state=42)

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(13,)),  # Adjust the input shape to match the number of MFCCs
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Save the model
model.save('custom_deepfake_detection_model.h5')
