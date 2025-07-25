import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# ------------------------------------------------------------------
# Portable paths – works from anywhere if saved_model/ is beside this file
# ------------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))        # directory of predict.py
MODEL_DIR  = os.path.join(BASE_DIR, "saved_model")             # ./saved_model
CLASS_FILE = os.path.join(MODEL_DIR, "class_indices.npy")

# ------------------------------------------------------------------
# Load model + label map once (thread-safe for Flask)
# ------------------------------------------------------------------
model = tf.saved_model.load(MODEL_DIR)
class_indices = np.load(CLASS_FILE, allow_pickle=True).item()
index_to_class = {v: k for k, v in class_indices.items()}

# ------------------------------------------------------------------
# Pre-processing identical to training
# ------------------------------------------------------------------
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    arr = image.img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0).astype(np.float32)

# ------------------------------------------------------------------
# Predict
# ------------------------------------------------------------------
def predict_image(img_path):
    X = preprocess_image(img_path)

    # SavedModel with serving_default signature
    infer  = model.signatures["serving_default"]
    logits = infer(tf.constant(X))[list(infer.structured_outputs.keys())[0]]
    probs  = tf.nn.softmax(logits).numpy()[0]
    idx    = int(np.argmax(probs))
    return index_to_class[idx], float(probs[idx])

# ------------------------------------------------------------------
# Quick CLI test (optional)
# ------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python predict.py path/to/image")
        sys.exit(1)
    label, conf = predict_image(sys.argv[1])
    print(f"{label} ({conf:.2%})")