import os
import sys

# Force TensorFlow to use legacy Keras 2 instead of Keras 3
# This ensures TensorFlow.js compatibility
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import tensorflowjs as tfjs
import tensorflow as tf

# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'saved_model.h5')
OUTPUT_DIR = os.path.join(BASE_DIR, '../public/model')

def convert():
    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found at {MODEL_PATH}. Please run train_model.py first.")
        return

    print(f"Converting model from {MODEL_PATH} to {OUTPUT_DIR}...")
    
    # Ensure output directory exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # Load the Keras model
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Convert to TFJS Layers format
    try:
        tfjs.converters.save_keras_model(model, OUTPUT_DIR)
        print(f"Successfully converted model to {OUTPUT_DIR}")
    except Exception as e:
        print(f"Error converting model: {e}")

if __name__ == '__main__':
    convert()

