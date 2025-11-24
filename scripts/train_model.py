import os
# Force TensorFlow to use legacy Keras 2 instead of Keras 3
# This ensures TensorFlow.js compatibility
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import json

# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '../src/app/images')
OUTPUT_DIR = os.path.join(BASE_DIR, '../public/model')
DOC_DIR = os.path.join(DATA_DIR, 'document')
REGULAR_DIR = os.path.join(DATA_DIR, 'regular_iamge') # Preserving typo from JS source

IMAGE_SIZE = 224
NUM_CLASSES = 2
EPOCHS = 100
BATCH_SIZE = 4
LEARNING_RATE = 0.00005
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'saved_model.h5')

def load_images():
    images = []
    labels = []

    # Load document images (label 0)
    if os.path.exists(DOC_DIR):
        doc_files = [f for f in os.listdir(DOC_DIR) if f.lower().endswith('.jpg')]
        print(f"Found {len(doc_files)} document images")
        for file in doc_files:
            img_path = os.path.join(DOC_DIR, file)
            try:
                img = load_img(img_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
                img_array = img_to_array(img)
                img_array = img_array / 255.0
                images.append(img_array)
                labels.append(0)
            except Exception as e:
                print(f"Error loading {file}: {e}")

    # Load regular images (label 1)
    if os.path.exists(REGULAR_DIR):
        reg_files = [f for f in os.listdir(REGULAR_DIR) if f.lower().endswith('.jpg')]
        print(f"Found {len(reg_files)} regular images")
        for file in reg_files:
            img_path = os.path.join(REGULAR_DIR, file)
            try:
                img = load_img(img_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
                img_array = img_to_array(img)
                img_array = img_array / 255.0
                images.append(img_array)
                labels.append(1)
            except Exception as e:
                print(f"Error loading {file}: {e}")
    
    if not images:
        return np.array([]), np.array([])
        
    return np.array(images), tf.keras.utils.to_categorical(np.array(labels), NUM_CLASSES)

def train():
    # Ensure output directory exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print("Loading images...")
    X, y = load_images()
    
    if len(X) == 0:
        print("No images found. Exiting.")
        return

    print(f"Data loaded: {len(X)} images")
    
    # Load MobileNet v1 with alpha 0.25 using tf.keras.applications
    print("Loading MobileNet...")
    base_model = tf.keras.applications.MobileNet(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        alpha=0.25,
        depth_multiplier=1,
        dropout=0.001,
        include_top=False,
        weights='imagenet',
        pooling=None
    )
    
    # Create new model using tf.keras
    inputs = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    # We need to make sure the base_model is running in inference mode
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(100, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = tf.keras.models.Model(inputs, outputs)
    
    # Freeze the MobileNet layers
    base_model.trainable = False
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Training model...")
    model.fit(
        X, y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        validation_split=0.0
    )
    
    print(f"Saving Keras model to {MODEL_SAVE_PATH}...")
    model.save(MODEL_SAVE_PATH)
    
    # Save metadata.json (same as JS script)
    metadata = {
        'labels': ['document', 'regular']
    }
    
    metadata_path = os.path.join(OUTPUT_DIR, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {metadata_path}")

if __name__ == '__main__':
    train()
