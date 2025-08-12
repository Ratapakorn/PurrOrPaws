# predict_keras.py
import sys, pathlib
from tensorflow import keras
import numpy as np

MODEL_PATH = "cats_dogs_final.keras"   # or .h5 if that's what you saved
THRESHOLD_FALLBACK = 0.410             # your tuned threshold

def load_threshold():
    p = pathlib.Path("cats_dogs_threshold.txt")
    if p.exists():
        try:
            return float(p.read_text().strip())
        except Exception:
            pass
    return THRESHOLD_FALLBACK

def main(paths):
    if not paths:
        print("Usage: python predict_keras.py <image1> [image2 ...]")
        return

    model = keras.models.load_model(MODEL_PATH, compile=False)
    t = load_threshold()

    # Infer input size from the model (e.g., 224x224x3)
    inp = model.inputs[0].shape
    H, W = int(inp[1]), int(inp[2])
    sigmoid_output = (int(model.outputs[0].shape[-1]) == 1)

    print(f"Loaded model: {MODEL_PATH}  | input={H}x{W}x3  | output={'sigmoid-1' if sigmoid_output else 'softmax-2'}")
    print(f"Using threshold t={t}")

    for path in paths:
        # Load & size the image
        img = keras.utils.load_img(path, target_size=(H, W))
        x = keras.utils.img_to_array(img)[None, ...]  # [1,H,W,3], raw [0..255]

        # IMPORTANT:
        # Your training pipeline already included mobilenet_v2.preprocess_input
        # *inside* the model graph. So we DO NOT preprocess here.
        # If you’re using an older model WITHOUT that layer, uncomment:
        # from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        # x = preprocess_input(x)

        y = model.predict(x, verbose=0)
        if sigmoid_output:
            p_dog = float(y[0][0])       # probability of "dog" (label=1)
        else:
            p_dog = float(y[0][1])       # softmax: [p_cat, p_dog]

        cls = "dog" if p_dog >= t else "cat"
        conf = p_dog if p_dog >= t else 1 - p_dog
        print(f"{path}: {cls} (pDog={p_dog:.3f}, t={t:.3f}, confidence={conf:.2f})")

if __name__ == "__main__":
    main(sys.argv[1:])
