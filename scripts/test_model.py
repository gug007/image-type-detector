import os
# Force TensorFlow to use legacy Keras 2 instead of Keras 3
# This ensures TensorFlow.js compatibility and consistency with training
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'saved_model.h5')
DATA_DIR = os.path.join(BASE_DIR, '../src/app/images')
DOC_DIR = os.path.join(DATA_DIR, 'document')
REGULAR_DIR = os.path.join(DATA_DIR, 'regular_iamge')

IMAGE_SIZE = 224
CLASSES = ['document', 'regular']

def predict_image(model, img_path):
    try:
        img = load_img(img_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
        img_array = img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        prediction = model.predict(img_array, verbose=0)
        class_idx = np.argmax(prediction[0])
        confidence = prediction[0][class_idx]
        
        return class_idx, confidence, prediction[0]
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None, None, None

def test_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}")
        return

    print(f"Loading model from {MODEL_PATH}...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    total_images = 0
    correct_predictions = 0
    
    print("\nTesting Document Images (Expected: document)...")
    if os.path.exists(DOC_DIR):
        files = [f for f in os.listdir(DOC_DIR) if f.lower().endswith('.jpg')]
        for f in files:
            path = os.path.join(DOC_DIR, f)
            pred_idx, conf, raw_pred = predict_image(model, path)
            
            if pred_idx is not None:
                total_images += 1
                predicted_label = CLASSES[pred_idx]
                is_correct = (pred_idx == 0)
                if is_correct:
                    correct_predictions += 1
                
                status = "✓" if is_correct else "✗"
                print(f"[{status}] {f}: Predicted={predicted_label} ({conf:.2%})")
                if not is_correct:
                    print(f"    Raw probabilities: Document={raw_pred[0]:.4f}, Regular={raw_pred[1]:.4f}")

    print("\nTesting Regular Images (Expected: regular)...")
    if os.path.exists(REGULAR_DIR):
        files = [f for f in os.listdir(REGULAR_DIR) if f.lower().endswith('.jpg')]
        for f in files:
            path = os.path.join(REGULAR_DIR, f)
            pred_idx, conf, raw_pred = predict_image(model, path)
            
            if pred_idx is not None:
                total_images += 1
                predicted_label = CLASSES[pred_idx]
                is_correct = (pred_idx == 1)
                if is_correct:
                    correct_predictions += 1
                
                status = "✓" if is_correct else "✗"
                print(f"[{status}] {f}: Predicted={predicted_label} ({conf:.2%})")
                if not is_correct:
                    print(f"    Raw probabilities: Document={raw_pred[0]:.4f}, Regular={raw_pred[1]:.4f}")

    if total_images > 0:
        accuracy = (correct_predictions / total_images) * 100
        print(f"\nOverall Accuracy: {accuracy:.2f}% ({correct_predictions}/{total_images})")
    else:
        print("\nNo images found to test.")

if __name__ == "__main__":
    test_model()

