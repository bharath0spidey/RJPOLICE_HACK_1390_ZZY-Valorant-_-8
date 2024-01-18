import numpy as np
import tensorflow as tf
import cv2
from sklearn.model_selection import train_test_split

# Load video file using OpenCV
video_file_path = r'E:\audio.detection\test_videos\aassnaulhq.mp4'
cap = cv2.VideoCapture(video_file_path)

# Initialize an empty list to store video frames
frames = []

# Read video frames
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)

# Convert frames to NumPy array
video_data = np.array(frames)

# Close the video file
cap.release()

# Assuming you have labels for your data
labels = np.random.randint(2, size=len(video_data))

# Get dimensions of video frames
height, width, channels = video_data.shape[1:]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(video_data, labels, test_size=0.2, random_state=42)

# Build a more advanced model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(height, width, channels)),
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.2)

# Save the model
model.save('video_deepfake_detection_model.h5')
