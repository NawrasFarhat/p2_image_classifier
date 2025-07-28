import argparse
import numpy as np
import json
from PIL import Image
import tensorflow as tf

def process_image(image):
    """Resize and normalize the image."""
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.image.resize(image, (224, 224))
    image /= 255.0  
    return image.numpy()

def predict(image_path, model, top_k=5):
    """Predict the top_k classes from an image using a trained model."""
    image = Image.open(image_path).convert('RGB')  
    np_image = np.asarray(image)
    processed_image = process_image(np_image)
    processed_image = np.expand_dims(processed_image, axis=0)

    probs = model.predict(processed_image)[0]
    top_indices = np.argsort(probs)[-top_k:][::-1]
    top_probs = probs[top_indices]
    return top_probs, top_indices

def main():
    parser = argparse.ArgumentParser(description='Predict flower name from an image.')
    parser.add_argument('image_path', type=str, help='Path to image file.')
    parser.add_argument('model_path', type=str, help='Path to saved Keras model (.h5 file).')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K predictions.')
    parser.add_argument('--category_names', type=str, default='label_map.json', help='Path to JSON label map.')
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model_path)

    try:
        with open(args.category_names, 'r') as f:
            class_names = json.load(f)
    except FileNotFoundError:
        print("Warning: Label map file not found. Using raw class indices.")
        class_names = {}

    
    top_probs, top_indices = predict(args.image_path, model, top_k=args.top_k)

    print("\nTop Predictions:")
    for i, (prob, idx) in enumerate(zip(top_probs, top_indices), 1):
        label = class_names.get(str(idx), f"Class_{idx}")
        print(f"{i}. {label} — Probability: {prob:.4f}")

if __name__ == '__main__':
    main()
