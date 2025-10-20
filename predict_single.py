import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
import numpy as np
import json
from PIL import Image

def load_model():
    try:
        model = keras.models.load_model('chest_xray_model.keras')
        with open('class_names.json', 'r') as f:
            class_names = json.load(f)
        return model, class_names
    except FileNotFoundError:
        print("Error: Model not found. Run train_model.py first.")
        sys.exit(1)

def predict_image(image_path):
    model, class_names = load_model()
    
    img = Image.open(image_path).convert('RGB')
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array, verbose=0)[0]
    predicted_class = class_names[np.argmax(predictions)]
    confidence = predictions[np.argmax(predictions)]
    
    print(f"\nImage: {os.path.basename(image_path)}")
    print(f"Prediction: {predicted_class}")
    print(f"Confidence: {confidence:.2%}")
    
    print("\nAll probabilities:")
    for cls, prob in zip(class_names, predictions):
        print(f"  {cls:15s}: {prob:.2%}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_single.py <image_path>")
        sys.exit(1)
    
    if not os.path.exists(sys.argv[1]):
        print(f"Error: File not found: {sys.argv[1]}")
        sys.exit(1)
    
    predict_image(sys.argv[1])
