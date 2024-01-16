import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = tf.keras.models.load_model("C:\\Users\\bhara\\OneDrive\\Documents\\Desktop\\archive\\Dataset\\deepfake_detection_model.h5")

# Specify the path to the image you want to test
image_path ="E:\\Dataset\\Test\\Fake\\fake_22.jpg"

# Load and preprocess the image
img = image.load_img(image_path, target_size=(128, 128))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0  # Normalize the pixel values to be between 0 and 1

# Make a prediction
prediction = model.predict(img_array)

# The output is a probability between 0 and 1. You can set a threshold to classify as real or fake.
threshold = 0.5
if prediction[0][0] > threshold:
    print("Real")
else:
    print("Fake")
