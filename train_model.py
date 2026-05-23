import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import os

# ==========================================
# CONFIGURATION
# ==========================================
DATA_DIR = 'dataset'
IMG_SIZE = (224, 224) # MobileNetV2 expects 224x224
BATCH_SIZE = 32
EPOCHS = 10           # How many times to study the data

# 1. SETUP DATA GENERATORS (The "Augmentation" Magic)
# This creates new variations of your photos on the fly
train_datagen = ImageDataGenerator(
    rescale=1./255,         # Normalize pixel values
    rotation_range=20,      # Rotate head slightly
    width_shift_range=0.2,  # Move face left/right
    height_shift_range=0.2, # Move face up/down
    horizontal_flip=True,   # Mirror image
    fill_mode='nearest',
    validation_split=0.2    # Use 20% of data to test accuracy
)

print("Loading images...")

# Load Training Data (80%)
train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',    # Binary = Confident vs Nervous
    subset='training'
)

# Load Validation Data (20%)
val_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# 2. BUILD THE MODEL
# We download MobileNetV2 (pre-trained on 1000s of objects)
# include_top=False means we remove the last layer so we can add our own
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model so we don't destroy its pre-learned logic
base_model.trainable = False

# Add our Custom "Confidence" Layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x) # Drop 50% of neurons to prevent memorization (Overfitting)
predictions = Dense(1, activation='sigmoid')(x) # Output layer (0 to 1)

model = Model(inputs=base_model.input, outputs=predictions)

# 3. COMPILE
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 4. TRAIN
print("\nStarting Training... (This may take a few minutes)")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator
)

# 5. SAVE THE BRAIN
model.save('my_confidence_model.h5')
print("\nSUCCESS! Model saved as 'my_confidence_model.h5'")
print("Class Indices:", train_generator.class_indices)