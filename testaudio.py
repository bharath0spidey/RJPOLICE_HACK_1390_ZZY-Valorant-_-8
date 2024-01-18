import numpy as np
import tensorflow as tf
import librosa

# Load the saved model
model = tf.keras.models.load_model('custom_deepfake_detection_model.h5')
# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=( 25077,)),  # Adjust the input shape to match the number of features in your MFCCs
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])


# Load data from an MP3 file using Librosa
new_mp3_file_path = r'E:\audio.detection\compressed\audio.mp3'  # Replace with the path to your new audio file
new_y, new_sr = librosa.load(new_mp3_file_path)

# Extract features (you may need to adjust these based on your requirements)
new_mfccs = librosa.feature.mfcc(y=new_y, sr=new_sr, n_mfcc=13)

# Reshape the data to match the model's input shape
X_test_reshaped = new_mfccs.T.reshape(1, -1)

# Make a prediction
prediction = model.predict(X_test_reshaped)

# Convert the prediction to a binary result
result = 'Real' if prediction[0][0] < 0.5 else 'Fake'

print(f'The audio is predicted as: {result}')