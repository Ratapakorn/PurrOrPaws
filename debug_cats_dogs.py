import pathlib, sys, random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

DATA_DIR = pathlib.Path("dataset")
CLASS_NAMES = ["cats", "dogs"]   # lock mapping 0=cats,1=dogs
IMG_SIZE = (224, 224)
BATCH = 8
SEED = 123
VALID_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}

def count_images(root: pathlib.Path):
    return {
        "cats": sum(1 for p in (root/"cats").rglob("*") if p.is_file() and p.suffix.lower() in VALID_EXTS),
        "dogs": sum(1 for p in (root/"dogs").rglob("*") if p.is_file() and p.suffix.lower() in VALID_EXTS),
    }

def make_ds(split="training"):
    # build first WITHOUT ignore_errors so we can read class_names
    base = keras.preprocessing.image_dataset_from_directory(
        DATA_DIR,
        labels="inferred",
        class_names=CLASS_NAMES,
        validation_split=0.2,
        subset=split,
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH,
        shuffle=True,
    )
    class_names = base.class_names
    # now add ignore_errors (new TF API) and prefetch
    ds = base.ignore_errors().prefetch(tf.data.AUTOTUNE)
    return ds, class_names

def peek_labels(ds, k=8):
    print("\nPeeking at a few samples (label: 0=cats, 1=dogs):")
    for i, (x, y) in enumerate(ds.unbatch().take(k)):
        print(f"  sample {i}: label={int(y.numpy())}")
    print()

def overfit_probe(take_per_class=12, epochs=20):
    # Build a tiny balanced set by file paths (bypasses directory label inference)
    cat_files = [p for p in (DATA_DIR/"cats").rglob("*") if p.suffix.lower() in VALID_EXTS]
    dog_files = [p for p in (DATA_DIR/"dogs").rglob("*") if p.suffix.lower() in VALID_EXTS]
    random.seed(SEED); random.shuffle(cat_files); random.shuffle(dog_files)
    cat_files, dog_files = cat_files[:take_per_class], dog_files[:take_per_class]
    if len(cat_files) < take_per_class or len(dog_files) < take_per_class:
        print(f"Not enough images for probe (need {take_per_class} per class)."); return None

    paths = [str(p) for p in cat_files] + [str(p) for p in dog_files]
    labels = [0]*len(cat_files) + [1]*len(dog_files)

    def load(path, label):
        img = tf.io.read_file(path)
        img = tf.io.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, IMG_SIZE)
        img = keras.applications.mobilenet_v2.preprocess_input(img)
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).shuffle(100, seed=SEED).batch(4).repeat()

    base = keras.applications.MobileNetV2(input_shape=IMG_SIZE+(3,), include_top=False, weights=None)
    model = keras.Sequential([
        layers.Input(shape=IMG_SIZE+(3,)),
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    steps = max(1, (2*take_per_class)//4)
    hist = model.fit(ds, steps_per_epoch=steps, epochs=epochs, verbose=0)
    last_acc = float(hist.history["accuracy"][-1])
    print(f"Overfit probe — last training accuracy: {last_acc:.3f}")
    return last_acc

def main():
    print("== Dataset counts ==")
    counts = count_images(DATA_DIR)
    print(f"  cats: {counts['cats']}")
    print(f"  dogs: {counts['dogs']}")
    if min(counts.values()) == 0:
        print("One class has 0 images. Fix dataset first."); sys.exit(1)

    ds_train, class_names = make_ds("training")
    print("\nclass_names reported by loader:", class_names)  # expect ['cats','dogs']
    peek_labels(ds_train, k=8)

    acc = overfit_probe(take_per_class=12, epochs=20)
    if acc is None: return
    print("\n== Interpretation ==")
    if acc >= 0.95:
        print("- Probe OK (≥0.95). Labels & preprocessing look fine. We can tune training next.")
    else:
        print("- Probe low (<0.95). Indicates a pipeline issue we should fix before full training.")

if __name__ == "__main__":
    main()
