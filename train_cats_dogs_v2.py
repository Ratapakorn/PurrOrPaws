# train_cats_dogs_v3.py
# Stratified splits + robust loading (JPEG/PNG only) + threshold tuning + reports

import os, pathlib, random, math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# --------------------
# Config
# --------------------
DATA_DIR = pathlib.Path("dataset")         # expects dataset/cats and dataset/dogs
CLASS_NAMES = ["cats", "dogs"]             # 0=cats, 1=dogs
VALID_EXTS = {".jpg", ".jpeg", ".png"}     # keep it simple/reliable
IMG_SIZE = (224, 224)
BATCH = 32
SEED = 1337

# Train schedule (good defaults for ~25k images total)
EPOCHS_HEAD = 6
EPOCHS_FT = 20
LR_HEAD = 1e-3
LR_FT = 2e-5
FREEZE_UP_TO = -90  # unfreeze last ~90 layers

AUTOTUNE = tf.data.AUTOTUNE
rng = random.Random(SEED)

# --------------------
# Helpers: collect files, stratified split per class
# --------------------
def list_images(folder: pathlib.Path):
    return [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in VALID_EXTS]

def stratified_splits():
    cats = list_images(DATA_DIR / "cats")
    dogs = list_images(DATA_DIR / "dogs")
    if len(cats) == 0 or len(dogs) == 0:
        raise RuntimeError("Need images in both dataset/cats and dataset/dogs.")
    for li in (cats, dogs):
        rng.shuffle(li)

    def split(lst):
        n = len(lst)
        n_train = int(0.8 * n)
        n_val   = int(0.1 * n)
        train = lst[:n_train]
        val   = lst[n_train:n_train+n_val]
        test  = lst[n_train+n_val:]
        return train, val, test

    c_tr, c_va, c_te = split(cats)
    d_tr, d_va, d_te = split(dogs)

    def labelize(paths, label):  # 0 cat, 1 dog
        return [(str(p), label) for p in paths]

    train = labelize(c_tr, 0) + labelize(d_tr, 1)
    val   = labelize(c_va, 0) + labelize(d_va, 1)
    test  = labelize(c_te, 0) + labelize(d_te, 1)
    rng.shuffle(train); rng.shuffle(val); rng.shuffle(test)

    print(f"Counts -> train cats:{len(c_tr)} dogs:{len(d_tr)} | "
          f"val cats:{len(c_va)} dogs:{len(d_va)} | "
          f"test cats:{len(c_te)} dogs:{len(d_te)}")
    return train, val, test

# --------------------
# tf.data pipeline (robust decode)
# --------------------
def decode_and_resize(path, label):
    img = tf.io.read_file(path)
    # decode_image supports jpeg/png/gif/bmp; we only pass jpeg/png here
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32)
    # Don't rescale here; use mobilenet_v2.preprocess_input later in the model
    return img, tf.cast(label, tf.int32)

def make_ds(pairs, training=False):
    paths = [p for p, _ in pairs]
    labels = [l for _, l in pairs]
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(buffer_size=len(paths), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(decode_and_resize, num_parallel_calls=AUTOTUNE)
    # If any file is unreadable/corrupt, skip it instead of crashing
    if hasattr(ds, "ignore_errors"):
        ds = ds.ignore_errors()
    else:
        ds = ds.apply(tf.data.experimental.ignore_errors())
    ds = ds.batch(BATCH).prefetch(AUTOTUNE)
    return ds

# --------------------
# Model
# --------------------
def build_model():
    base = keras.applications.MobileNetV2(input_shape=IMG_SIZE + (3,),
                                          include_top=False, weights="imagenet")
    base.trainable = False
    inputs = keras.Input(shape=IMG_SIZE + (3,))
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.05)(x)
    x = layers.RandomZoom(0.1)(x)
    x = keras.applications.mobilenet_v2.preprocess_input(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    return model, base

# --------------------
# Metrics / threshold tuning
# --------------------
def collect_probs_and_labels(model, ds):
    y_true, y_prob = [], []
    for x, y in ds:
        p = model.predict(x, verbose=0).ravel()
        y_prob.append(p)
        y_true.append(y.numpy().ravel())
    return np.concatenate(y_true), np.concatenate(y_prob)

def best_threshold(y_true, y_prob):
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.2, 0.8, 121):
        y_pred = (y_prob >= t).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        prec = tp / (tp + fp + 1e-9)
        rec  = tp / (tp + fn + 1e-9)
        f1   = 2*prec*rec / (prec + rec + 1e-9)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1

def report(y_true, y_prob, t, title):
    y_pred = (y_prob >= t).astype(int)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2*prec*rec / (prec + rec + 1e-9)
    acc  = (tp + tn) / max(1, len(y_true))
    print(f"\n== {title} @ t={t:.3f} ==")
    print(f"[[TN={tn}, FP={fp}], [FN={fn}, TP={tp}]]")
    print(f"acc={acc:.3f}  prec={prec:.3f}  rec={rec:.3f}  f1={f1:.3f}")

# --------------------
# Train + evaluate
# --------------------
def main():
    train_pairs, val_pairs, test_pairs = stratified_splits()
    train_ds = make_ds(train_pairs, training=True)
    val_ds   = make_ds(val_pairs, training=False)
    test_ds  = make_ds(test_pairs, training=False)

    print("Sanity — each split has both classes:",
          {k: sum(1 for _, y in s) for k, s in
           [("train_cats", [(p,y) for p,y in train_pairs if y==0]),
            ("train_dogs", [(p,y) for p,y in train_pairs if y==1]),
            ("val_cats",   [(p,y) for p,y in val_pairs if y==0]),
            ("val_dogs",   [(p,y) for p,y in val_pairs if y==1]),
            ("test_cats",  [(p,y) for p,y in test_pairs if y==0]),
            ("test_dogs",  [(p,y) for p,y in test_pairs if y==1]) ]})

    # Optional class weights (helps if imbalance happens after filtering)
    n_cats = sum(1 for _, y in train_pairs if y == 0)
    n_dogs = sum(1 for _, y in train_pairs if y == 1)
    total  = max(1, n_cats + n_dogs)
    class_weight = {0: total/(2*max(1, n_cats)), 1: total/(2*max(1, n_dogs))}
    print("Class weights:", class_weight)

    model, base = build_model()
    model.compile(optimizer=keras.optimizers.Adam(LR_HEAD),
                  loss="binary_crossentropy", metrics=["accuracy"])

    # Head training
    model.fit(train_ds, validation_data=val_ds,
              epochs=EPOCHS_HEAD,
              class_weight=class_weight,
              callbacks=[
                  keras.callbacks.ModelCheckpoint("best_head.keras",
                                                  monitor="val_accuracy",
                                                  save_best_only=True),
                  keras.callbacks.EarlyStopping(monitor="val_accuracy",
                                                patience=3,
                                                restore_best_weights=True),
                  keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
                                                    factor=0.5,
                                                    patience=2,
                                                    min_lr=1e-6),
              ])

    # Fine-tune
    base.trainable = True
    if FREEZE_UP_TO != 0:
        for layer in base.layers[:FREEZE_UP_TO]:
            layer.trainable = False
    model.compile(optimizer=keras.optimizers.Adam(LR_FT),
                  loss="binary_crossentropy", metrics=["accuracy"])

    model.fit(train_ds, validation_data=val_ds,
              epochs=EPOCHS_FT,
              class_weight=class_weight,
              callbacks=[
                  keras.callbacks.ModelCheckpoint("best_finetuned.keras",
                                                  monitor="val_accuracy",
                                                  save_best_only=True),
                  keras.callbacks.EarlyStopping(monitor="val_accuracy",
                                                patience=4,
                                                restore_best_weights=True),
                  keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
                                                    factor=0.5,
                                                    patience=2,
                                                    min_lr=1e-6),
              ])

    # Threshold tuning on validation
    yv_true, yv_prob = collect_probs_and_labels(model, val_ds)
    t_best, f1_best = best_threshold(yv_true, yv_prob)
    print(f"\nBest threshold on validation: {t_best:.3f} (val F1={f1_best:.3f})")
    report(yv_true, yv_prob, t_best, "Validation")

    # Test report
    yt_true, yt_prob = collect_probs_and_labels(model, test_ds)
    report(yt_true, yt_prob, t_best, "Test")

    # Save model + threshold
    model.save("cats_dogs_final.keras")
    with open("cats_dogs_threshold.txt", "w") as f:
        f.write(str(t_best))
    print("\nSaved model -> cats_dogs_final.keras")
    print("Saved tuned threshold -> cats_dogs_threshold.txt")

    # Quick inference helper
    def predict_image(path):
        img = keras.utils.load_img(path, target_size=IMG_SIZE)
        x = keras.utils.img_to_array(img)[None, ...]
        x = keras.applications.mobilenet_v2.preprocess_input(x)
        p = float(model.predict(x, verbose=0)[0][0])
        t = t_best
        cls = "dog" if p >= t else "cat"
        conf = p if p >= t else (1 - p)
        print(f"{path}: {cls} (p={p:.3f}, t={t:.3f}, conf={conf:.2f})")

    # Example after training:
    # predict_image("dataset/cats/ginger-cat.jpg")
    # predict_image("dataset/dogs/Cute_dog.jpg")

if __name__ == "__main__":
    tf.get_logger().setLevel("ERROR")
    main()
