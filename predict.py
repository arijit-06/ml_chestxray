import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt

def load_model_and_classes():
    try:
        model = keras.models.load_model('chest_xray_model.keras')
        with open('class_names.json', 'r') as f:
            class_names = json.load(f)
        return model, class_names
    except FileNotFoundError:
        print("Error: Model files not found. Run train_model.py first.")
        sys.exit(1)

def preprocess_image(image_path, img_size=128):
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((img_size, img_size))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array, img
    except Exception as e:
        print(f"Error loading image: {e}")
        sys.exit(1)

def predict_disease(image_path, show_visualization=True):
    model, class_names = load_model_and_classes()
    img_array, original_img = preprocess_image(image_path)
    
    predictions = model.predict(img_array, verbose=0)[0]
    predicted_class_idx = np.argmax(predictions)
    predicted_class = class_names[predicted_class_idx]
    confidence = predictions[predicted_class_idx]
    
    print(f"\nDiagnosis: {predicted_class.upper()}")
    print(f"Confidence: {confidence:.2%}")
    
    print("\nAll Probabilities:")
    for cls, prob in zip(class_names, predictions):
        bar = '█' * int(prob * 40)
        print(f"  {cls:15s}: {prob:.2%} {bar}")
    
    if confidence > 0.90:
        print("\nHigh confidence prediction")
    elif confidence > 0.70:
        print("\nModerate confidence - consider further examination")
    else:
        print("\nLow confidence - expert review recommended")
    
    if show_visualization:
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.imshow(original_img, cmap='gray')
        plt.title(f'Input X-Ray\n{os.path.basename(image_path)}', fontweight='bold')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        colors = ['green' if i == predicted_class_idx else 'gray' for i in range(len(class_names))]
        plt.barh(class_names, predictions, color=colors)
        plt.xlabel('Probability', fontweight='bold')
        plt.title(f'Prediction: {predicted_class}\n{confidence:.1%} confidence', fontweight='bold')
        plt.xlim([0, 1])
        
        plt.tight_layout()
        plt.savefig('prediction_result.png', dpi=150, bbox_inches='tight')
        print("\nVisualization saved: prediction_result.png")
        plt.show()
    
    return predicted_class, confidence, predictions

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_xray_image>")
        print("Example: python predict.py ./chest_xray_data/test/PNEUMONIA/person1_virus_6.jpeg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        sys.exit(1)
    
    predicted_class, confidence, probabilities = predict_disease(image_path)
