import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import scipy


# Load your dataset - This is just a placeholder; replace it with your dataset.
# The dataset should have two folders: 'real' and 'fake'.
train_data_dir = "Train\\"
validation_data_dir = "Validation\\"

# Image parameters
img_width, img_height = 128, 128
batch_size = 32

# Create data generators with augmentation for better generalizationj
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)

validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',  # 'binary' for a two-class problem (real or fake)
)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
)

# Define the CNN model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), input_shape=(img_width, img_height, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

# Compile the model
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy'],
)

# Train the model
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=10,  # You may need more epochs depending on your dataset
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
)

# Save the model
model.save('deepfake_detection_model.h5')
