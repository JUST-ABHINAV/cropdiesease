import os
import shutil
import numpy as np
import joblib
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Import our feature extraction module (must be in the same folder)
from features import extract_features, FEATURE_LEN

print(f"Feature vector length: {FEATURE_LEN}")
print("All imports OK.")


# ── Cell 2: Download & flatten dataset ────────────────────────────────────────
import kagglehub

path = kagglehub.dataset_download("alfiyasiddique/indian-crop-disease-dataset")
CROP_ROOT    = os.path.join(path, "Crop Dataset")
DATASET_FLAT = "/content/dataset_flat"
os.makedirs(DATASET_FLAT, exist_ok=True)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

for crop in sorted(os.listdir(CROP_ROOT)):
    crop_path = os.path.join(CROP_ROOT, crop)
    if not os.path.isdir(crop_path):
        continue
    for split in os.listdir(crop_path):
        split_path = os.path.join(crop_path, split)
        if not os.path.isdir(split_path):
            continue
        if split.lower() == "healthy":
            dst = os.path.join(DATASET_FLAT, f"{crop}_healthy")
            os.makedirs(dst, exist_ok=True)
            for img in os.listdir(split_path):
                if os.path.splitext(img)[1].lower() in IMG_EXTS:
                    shutil.copy(os.path.join(split_path, img), dst)
        elif split.lower() == "disease":
            for disease in os.listdir(split_path):
                dis_path = os.path.join(split_path, disease)
                if not os.path.isdir(dis_path):
                    continue
                cls_name = f"{crop}_{disease.lower().replace(' ', '_')}"
                dst = os.path.join(DATASET_FLAT, cls_name)
                os.makedirs(dst, exist_ok=True)
                for img in os.listdir(dis_path):
                    if os.path.splitext(img)[1].lower() in IMG_EXTS:
                        shutil.copy(os.path.join(dis_path, img), dst)

# Print summary
print("Flattened classes:")
total = 0
for cls in sorted(os.listdir(DATASET_FLAT)):
    n = len(os.listdir(os.path.join(DATASET_FLAT, cls)))
    print(f"  {cls:<45} {n} images")
    total += n
print(f"\nTotal: {len(os.listdir(DATASET_FLAT))} classes, {total} images")


# ── Cell 3: Core helper — load images as feature vectors ──────────────────────

def load_features_and_labels(data_dir, label_map=None, verbose=True):
    """
    Walk through data_dir, extract features from every image,
    and return X (feature matrix) and y (label array).

    Parameters
    ----------
    data_dir  : str   Path to a folder where each subfolder = one class
    label_map : dict  Optional {folder_name: integer_label}
                      If None, labels are assigned alphabetically.
    verbose   : bool  Print progress

    Returns
    -------
    X          : np.ndarray  shape (n_images, FEATURE_LEN)
    y          : np.ndarray  shape (n_images,) integer labels
    class_names: list of str  class_names[i] corresponds to label i
    """
    class_folders = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])

    if label_map is None:
        label_map = {name: idx for idx, name in enumerate(class_folders)}

    class_names = [None] * len(label_map)
    for name, idx in label_map.items():
        class_names[idx] = name

    X_list, y_list = [], []
    errors = 0

    for folder in class_folders:
        folder_path = os.path.join(data_dir, folder)
        label       = label_map[folder]
        images      = [f for f in os.listdir(folder_path)
                       if os.path.splitext(f)[1].lower() in IMG_EXTS]

        if verbose:
            print(f"  [{folder}]  {len(images)} images ...", end="", flush=True)

        folder_feats = []
        for img_file in images:
            img_path = os.path.join(folder_path, img_file)
            try:
                img  = Image.open(img_path)
                feat = extract_features(img)
                folder_feats.append(feat)
                y_list.append(label)
            except Exception as e:
                errors += 1   # skip corrupt images silently

        X_list.extend(folder_feats)
        if verbose:
            print(f" done ({len(folder_feats)} extracted)")

    if errors:
        print(f"  Warning: {errors} images could not be read and were skipped.")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    return X, y, class_names


# ── Cell 4: Train and evaluate one model ──────────────────────────────────────

def train_and_evaluate(X, y, class_names, model_name="model"):
    """
    Split data, train a Random Forest, print a full evaluation report.

    Random Forest is ideal here because:
    - Handles small, imbalanced datasets well
    - class_weight='balanced' compensates for class imbalance automatically
    - No feature scaling required (unlike SVM or logistic regression)
    - Fast to train and predict
    - Built-in predict_proba() which the app needs for confidence scores

    Returns trained model (fitted on full training + validation data).
    """
    # ── 70/15/15 stratified split ─────────────────────────────────────────────
    # stratify=y ensures every class appears in all three splits proportionally
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.15 / 0.85, stratify=y_tv, random_state=42
    )

    print(f"  Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

    # ── Build classifier ──────────────────────────────────────────────────────
    clf = RandomForestClassifier(
        n_estimators=200,          # 200 decision trees in the forest
        max_depth=None,            # trees grow until pure leaves
        min_samples_split=4,       # minimum samples to split a node
        min_samples_leaf=2,        # minimum samples in a leaf
        class_weight="balanced",   # auto-adjust for imbalanced classes
                                   # (rare classes get higher weight)
        n_jobs=-1,                 # use all CPU cores
        random_state=42,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    print("  Training...", end="", flush=True)
    clf.fit(X_train, y_train)
    print(" done.")

    # ── Validation accuracy ───────────────────────────────────────────────────
    val_preds = clf.predict(X_val)
    val_acc   = accuracy_score(y_val, val_preds)
    print(f"  Validation accuracy : {val_acc * 100:.2f}%")

    # ── Test accuracy & per-class report ─────────────────────────────────────
    test_preds = clf.predict(X_test)
    test_acc   = accuracy_score(y_test, test_preds)
    print(f"  Test accuracy       : {test_acc * 100:.2f}%")
    print()
    print(classification_report(y_test, test_preds, target_names=class_names, digits=3))

    # ── Retrain on full train+val data for best final model ───────────────────
    # After evaluation, use all available labelled data for the saved model
    clf.fit(np.vstack([X_train, X_val]), np.concatenate([y_train, y_val]))
    print(f"  Final model retrained on train+val ({len(X_train)+len(X_val)} samples).")

    return clf


# ── Cell 5: Stage 1 — Crop Classifier ─────────────────────────────────────────

CROPS = ["coffee", "cotton", "jute", "rice", "sugarcane", "wheat"]

# Build crop-level dataset:
# Pool all images from each crop regardless of disease into one crop folder
CROP_DS = "/content/dataset_crop"
os.makedirs(CROP_DS, exist_ok=True)

for cls_folder in os.listdir(DATASET_FLAT):
    cls_path = os.path.join(DATASET_FLAT, cls_folder)
    if not os.path.isdir(cls_path):
        continue
    crop = next((c for c in CROPS if cls_folder.startswith(c)), None)
    if crop is None:
        continue
    dst = os.path.join(CROP_DS, crop)
    os.makedirs(dst, exist_ok=True)
    for img in os.listdir(cls_path):
        if os.path.splitext(img)[1].lower() in IMG_EXTS:
            src      = os.path.join(cls_path, img)
            dst_file = os.path.join(dst, f"{cls_folder}__{img}")
            shutil.copy(src, dst_file)

print("Crop-level dataset:")
for crop in sorted(os.listdir(CROP_DS)):
    n = len(os.listdir(os.path.join(CROP_DS, crop)))
    print(f"  {crop:<15} {n} images")

print("\n=== STAGE 1: Training Crop Classifier ===")
X_crop, y_crop, crop_classes = load_features_and_labels(CROP_DS)
print(f"\nFeature matrix shape: {X_crop.shape}")

crop_model = train_and_evaluate(X_crop, y_crop, crop_classes, "crop_model")

# Save
joblib.dump(crop_model, "crop_model.pkl")
with open("crop_classes.txt", "w") as f:
    f.write("\n".join(crop_classes))
print("Saved: crop_model.pkl  crop_classes.txt")


# ── Cell 6: Stage 2 — One Disease Classifier Per Crop ─────────────────────────

DISEASE_ROOT = "/content/disease_per_crop"

# Build per-crop disease folders
for crop in CROPS:
    crop_disease_dir = os.path.join(DISEASE_ROOT, crop)
    os.makedirs(crop_disease_dir, exist_ok=True)
    for cls_folder in sorted(os.listdir(DATASET_FLAT)):
        if not cls_folder.startswith(crop):
            continue
        src = os.path.join(DATASET_FLAT, cls_folder)
        dst = os.path.join(crop_disease_dir, cls_folder)
        if not os.path.exists(dst):
            shutil.copytree(src, dst)

print("Per-crop disease folders ready.")

# Train one model per crop
for crop in CROPS:
    print(f"\n{'='*55}")
    print(f"  STAGE 2 — {crop.upper()} Disease Classifier")
    print(f"{'='*55}")

    crop_data_dir = os.path.join(DISEASE_ROOT, crop)
    X_d, y_d, d_classes = load_features_and_labels(crop_data_dir)
    print(f"  Feature matrix shape: {X_d.shape}")
    print(f"  Classes: {d_classes}")

    d_model = train_and_evaluate(X_d, y_d, d_classes, f"{crop}_model")

    # Save model and class names
    joblib.dump(d_model, f"{crop}_model.pkl")
    with open(f"{crop}_classes.txt", "w") as f:
        f.write("\n".join(d_classes))
    print(f"  Saved: {crop}_model.pkl  {crop}_classes.txt")

print("\n✅ All models trained and saved!")


# ── Cell 8: Quick prediction test ─────────────────────────────────────────────

import random

def predict_two_stage_test(img_path, crop_model, crop_classes, disease_models, disease_classes):
    """
    Standalone test of the two-stage pipeline.
    Mirrors exactly what app.py does.
    """
    img  = Image.open(img_path)
    feat = extract_features(img).reshape(1, -1)   # shape (1, 368)

    # Stage 1
    c_probs   = crop_model.predict_proba(feat)[0]   # shape (n_crops,)
    crop_idx  = int(np.argmax(c_probs))
    crop_name = crop_classes[crop_idx]
    crop_conf = float(c_probs[crop_idx]) * 100

    # Stage 2
    d_model   = disease_models[crop_name]
    d_classes_list = disease_classes[crop_name]
    d_probs   = d_model.predict_proba(feat)[0]
    dis_idx   = int(np.argmax(d_probs))
    dis_name  = d_classes_list[dis_idx]
    dis_conf  = float(d_probs[dis_idx]) * 100

    return crop_name, crop_conf, dis_name, dis_conf

# Load all saved models fresh
crop_model_loaded = joblib.load("crop_model.pkl")
with open("crop_classes.txt") as f:
    crop_classes_loaded = [l.strip() for l in f if l.strip()]

disease_models_loaded, disease_classes_loaded = {}, {}
for c in CROPS:
    disease_models_loaded[c] = joblib.load(f"{c}_model.pkl")
    with open(f"{c}_classes.txt") as f:
        disease_classes_loaded[c] = [l.strip() for l in f if l.strip()]

# Pick a random image
rand_cls    = random.choice(os.listdir(DATASET_FLAT))
rand_folder = os.path.join(DATASET_FLAT, rand_cls)
rand_img    = random.choice(os.listdir(rand_folder))
rand_path   = os.path.join(rand_folder, rand_img)

crop_name, crop_conf, dis_name, dis_conf = predict_two_stage_test(
    rand_path, crop_model_loaded, crop_classes_loaded,
    disease_models_loaded, disease_classes_loaded
)

print(f"\nTest image : {rand_cls}/{rand_img}")
print(f"True class : {rand_cls}")
print(f"Stage 1    : {crop_name}  ({crop_conf:.1f}%)")
print(f"Stage 2    : {dis_name}  ({dis_conf:.1f}%)")
print(f"Correct    : {'✓' if rand_cls == dis_name else '✗'}")
