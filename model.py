import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
IMAGE_SIZE  = (128, 128)
BATCH_SIZE  = 32
EPOCHS      = 50
DATASET_DIR = r"put dataset folder path with several categories here"
MODEL_DIR   = r"just define a path to saved_model folder"
os.makedirs(MODEL_DIR, exist_ok=True)

# ------------------------------------------------------------------
# Stronger augmentation
# ------------------------------------------------------------------
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True
)

train_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

num_classes = len(train_gen.class_indices)

# ------------------------------------------------------------------
# Deeper CNN (unchanged)
# ------------------------------------------------------------------
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(*IMAGE_SIZE, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# ------------------------------------------------------------------
# Lower initial LR + tuned callbacks
# ------------------------------------------------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_accuracy', factor=0.3, patience=4, min_lr=1e-6, verbose=1)
]

model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    callbacks=callbacks
)

# ------------------------------------------------------------------
# Export
# ------------------------------------------------------------------
model.export(MODEL_DIR)
np.save(os.path.join(MODEL_DIR, 'class_indices.npy'), train_gen.class_indices)

print("Training complete.")
print("Model exported to:", MODEL_DIR)
print("Class indices:", train_gen.class_indices)