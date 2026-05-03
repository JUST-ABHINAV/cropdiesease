import os  # paths and directory listing
import shutil  # copy files / trees
import random  # random smoke-test image picker
import numpy as np  # stacked feature matrices and arrays
import joblib  # save/load sklearn models
from PIL import Image  # open RGB images from disk
from sklearn.ensemble import RandomForestClassifier  # classifier
from sklearn.model_selection import train_test_split  # stratified splits
from sklearn.metrics import classification_report, accuracy_score  # evaluation
from sklearn.preprocessing import LabelEncoder  # not used below

from features import extract_features, FEATURE_LEN  # 368-D feature pipeline

print(f"Feature vector length: {FEATURE_LEN}")  # confirm constants
print("All imports OK.")  # startup banner

import kagglehub  # downloaded after lightweight imports (notebook-style ordering)

path = kagglehub.dataset_download("alfiyasiddique/indian-crop-disease-dataset")  # local cache path
CROP_ROOT = os.path.join(path, "Crop Dataset")  # nested crop splits from hub
DATASET_FLAT = "/content/dataset_flat"  # one folder per flattened class label
os.makedirs(DATASET_FLAT, exist_ok=True)  # ensure output root exists

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}  # accepted image suffixes

for crop in sorted(os.listdir(CROP_ROOT)):  # each crop folder (e.g. rice)
    crop_path = os.path.join(CROP_ROOT, crop)  # full path to crop
    if not os.path.isdir(crop_path):  # skip stray files
        continue
    for split in os.listdir(crop_path):  # typically healthy / disease
        split_path = os.path.join(crop_path, split)  # path to split
        if not os.path.isdir(split_path):  # skip non-directories
            continue
        if split.lower() == "healthy":  # merge all healthy into one label
            dst = os.path.join(DATASET_FLAT, f"{crop}_healthy")  # flat class name
            os.makedirs(dst, exist_ok=True)  # class output folder
            for img in os.listdir(split_path):  # each image file
                if os.path.splitext(img)[1].lower() in IMG_EXTS:  # filter by extension
                    shutil.copy(os.path.join(split_path, img), dst)  # copy into flat layout
        elif split.lower() == "disease":  # nested disease-type folders
            for disease in os.listdir(split_path):  # disease name folders
                dis_path = os.path.join(split_path, disease)  # path to disease images
                if not os.path.isdir(dis_path):  # skip files
                    continue
                cls_name = f"{crop}_{disease.lower().replace(' ', '_')}"  # stable folder name
                dst = os.path.join(DATASET_FLAT, cls_name)  # one folder per crop+disease
                os.makedirs(dst, exist_ok=True)  # create class folder
                for img in os.listdir(dis_path):  # iterate images
                    if os.path.splitext(img)[1].lower() in IMG_EXTS:  # image only
                        shutil.copy(os.path.join(dis_path, img), dst)  # into flat dataset

print("Flattened classes:")  # human-readable summary header
total = 0  # count all copied images
for cls in sorted(os.listdir(DATASET_FLAT)):  # each class subdirectory
    n = len(os.listdir(os.path.join(DATASET_FLAT, cls)))  # file count (not filtered by ext here)
    print(f"  {cls:<45} {n} images")  # print class size
    total += n  # accumulate total
print(f"\nTotal: {len(os.listdir(DATASET_FLAT))} classes, {total} images")  # dataset stats


def load_features_and_labels(data_dir, label_map=None, verbose=True):
    class_folders = sorted([  # alphanumeric order of labels
        d for d in os.listdir(data_dir)  # every entry in root
        if os.path.isdir(os.path.join(data_dir, d))  # keep folders only
    ])

    if label_map is None:  # default: alphabetical index assignment
        label_map = {name: idx for idx, name in enumerate(class_folders)}  # str -> int id

    class_names = [None] * len(label_map)  # index -> folder name array
    for name, idx in label_map.items():  # invert map into list order
        class_names[idx] = name  # slot idx holds class string

    X_list, y_list = [], []  # row-wise accumulation
    errors = 0  # count failed reads

    for folder in class_folders:  # one ML class per subfolder
        folder_path = os.path.join(data_dir, folder)  # absolute class path
        label = label_map[folder]  # integer label for this folder
        images = [f for f in os.listdir(folder_path)  # filenames in folder
                   if os.path.splitext(f)[1].lower() in IMG_EXTS]  # images only

        if verbose:  # progress line prefix
            print(f"  [{folder}]  {len(images)} images ...", end="", flush=True)

        folder_feats = []  # features for current class in visit order
        for img_file in images:  # each image filename
            img_path = os.path.join(folder_path, img_file)  # full image path
            try:
                img = Image.open(img_path)  # lazy load PIL image
                feat = extract_features(img)  # 368-D numpy vector
                folder_feats.append(feat)  # stack row later
                y_list.append(label)  # paired label sample
            except Exception:  # corrupt or unsupported decode
                errors += 1  # track skips

        X_list.extend(folder_feats)  # append all rows from this folder
        if verbose:  # finish progress line
            print(f" done ({len(folder_feats)} extracted)")  # extraction count

    if errors:  # report if anything failed
        print(f"  Warning: {errors} images could not be read and were skipped.")

    X = np.array(X_list, dtype=np.float32)  # (n_samples, FEATURE_LEN)
    y = np.array(y_list, dtype=np.int32)  # parallel labels
    return X, y, class_names  # training arrays + name table


def train_and_evaluate(X, y, class_names, model_name="model"):
    X_tv, X_test, y_tv, y_test = train_test_split(  # hold out 15% test
        X, y, test_size=0.15, stratify=y, random_state=42  # proportional classes
    )
    X_train, X_val, y_train, y_val = train_test_split(  # 15% of full -> val (~12.75% of all)
        X_tv, y_tv, test_size=0.15 / 0.85, stratify=y_tv, random_state=42  # stratified on remainder
    )

    print(f"  Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")  # split sizes

    clf = RandomForestClassifier(  # bagged trees
        n_estimators=200,  # number of trees
        max_depth=None,  # grow until split rules stop
        min_samples_split=4,  # min samples to attempt a split
        min_samples_leaf=2,  # min samples per leaf
        class_weight="balanced",  # inverse-frequency sample weights per class
        n_jobs=-1,  # parallel fit across CPUs
        random_state=42,  # reproducibility
    )

    print("Training...", end="", flush=True)  # no newline until done
    clf.fit(X_train, y_train)  # train on train split
    print(" done.")  # newline after fit

    val_preds = clf.predict(X_val)  # validation predictions
    val_acc = accuracy_score(y_val, val_preds)  # fraction correct on val
    print(f"  Validation accuracy : {val_acc * 100:.2f}%")  # pretty print percent

    test_preds = clf.predict(X_test)  # held-out test predictions
    test_acc = accuracy_score(y_test, test_preds)  # fraction correct on test
    print(f"  Test accuracy       : {test_acc * 100:.2f}%")  # pretty print percent
    print()  # blank line before report
    print(classification_report(y_test, test_preds, target_names=class_names, digits=3))  # per-class metrics

    clf.fit(np.vstack([X_train, X_val]), np.concatenate([y_train, y_val]))  # refit on train+val for deployment
    print(f"  Final model retrained on train+val ({len(X_train)+len(X_val)} samples).")  # note refit

    return clf  # fitted forest


CROPS = ["coffee", "cotton", "jute", "rice", "sugarcane", "wheat"]  # supported crop ids

CROP_DS = "/content/dataset_crop"  # pooled images: one folder per crop only
os.makedirs(CROP_DS, exist_ok=True)  # create crop-level root

for cls_folder in os.listdir(DATASET_FLAT):  # each fine-grained class folder
    cls_path = os.path.join(DATASET_FLAT, cls_folder)  # path to disease/fine folder
    if not os.path.isdir(cls_path):  # skip junk
        continue
    crop = next((c for c in CROPS if cls_folder.startswith(c)), None)  # match crop prefix on folder name
    if crop is None:  # skip unknown crops if any
        continue
    dst = os.path.join(CROP_DS, crop)  # coarse crop bucket
    os.makedirs(dst, exist_ok=True)  # ensure bucket exists
    for img in os.listdir(cls_path):  # all images in fine class
        if os.path.splitext(img)[1].lower() in IMG_EXTS:  # image files
            src = os.path.join(cls_path, img)  # source file
            dst_file = os.path.join(dst, f"{cls_folder}__{img}")  # unique name avoids overwrite
            shutil.copy(src, dst_file)  # duplicate into crop pool

print("Crop-level dataset:")  # section header
for crop in sorted(os.listdir(CROP_DS)):  # each crop folder
    n = len(os.listdir(os.path.join(CROP_DS, crop)))  # image count
    print(f"  {crop:<15} {n} images")  # print row

print("\n=== STAGE 1: Training Crop Classifier ===")  # stage banner
X_crop, y_crop, crop_classes = load_features_and_labels(CROP_DS)  # build design matrix
print(f"\nFeature matrix shape: {X_crop.shape}")  # confirm (n, 368)

crop_model = train_and_evaluate(X_crop, y_crop, crop_classes, "crop_model")  # train + eval + final refit

joblib.dump(crop_model, "crop_model.pkl")  # persist sklearn object
with open("crop_classes.txt", "w") as f:  # label order file for inference
    f.write("\n".join(crop_classes))  # one class name per line
print("Saved: crop_model.pkl  crop_classes.txt")  # confirm paths


DISEASE_ROOT = "/content/disease_per_crop"  # nested by crop then fine class

for crop in CROPS:  # build directory tree for stage-2 data
    crop_disease_dir = os.path.join(DISEASE_ROOT, crop)  # e.g. .../sugarcane
    os.makedirs(crop_disease_dir, exist_ok=True)  # ensure crop subroot
    for cls_folder in sorted(os.listdir(DATASET_FLAT)):  # scan flat classes
        if not cls_folder.startswith(crop):  # only classes for this crop
            continue
        src = os.path.join(DATASET_FLAT, cls_folder)  # source class folder
        dst = os.path.join(crop_disease_dir, cls_folder)  # mirror name under crop
        if not os.path.exists(dst):  # avoid re-copy errors
            shutil.copytree(src, dst)  # full folder copy

print("Per-crop disease folders ready.")  # dataset prep done

for crop in CROPS:  # train six separate disease classifiers
    print(f"\n{'='*55}")  # visual separator
    print(f"  STAGE 2 — {crop.upper()} Disease Classifier")  # which crop
    print(f"{'='*55}")  # close separator

    crop_data_dir = os.path.join(DISEASE_ROOT, crop)  # path to that crop's fine classes
    X_d, y_d, d_classes = load_features_and_labels(crop_data_dir)  # features for this crop only
    print(f"  Feature matrix shape: {X_d.shape}")  # shape check
    print(f"  Classes: {d_classes}")  # list of label strings

    d_model = train_and_evaluate(X_d, y_d, d_classes, f"{crop}_model")  # train disease RF

    joblib.dump(d_model, f"{crop}_model.pkl")  # save per-crop model
    with open(f"{crop}_classes.txt", "w") as f:  # class order for inference
        f.write("\n".join(d_classes))  # newline-separated names
    print(f"  Saved: {crop}_model.pkl  {crop}_classes.txt")  # confirm

print("\n✅ All models trained and saved!")  # all stages complete


def predict_two_stage_test(img_path, crop_model, crop_classes, disease_models, disease_classes):
    img = Image.open(img_path)  # load query image
    feat = extract_features(img).reshape(1, -1)  # single-sample matrix for sklearn API

    c_probs = crop_model.predict_proba(feat)[0]  # class probabilities over crops
    crop_idx = int(np.argmax(c_probs))  # top-1 crop index
    crop_name = crop_classes[crop_idx]  # crop string id
    crop_conf = float(c_probs[crop_idx]) * 100  # percentage confidence stage 1

    d_model = disease_models[crop_name]  # RF for predicted crop bucket
    d_classes_list = disease_classes[crop_name]  # fine labels for that crop
    d_probs = d_model.predict_proba(feat)[0]  # disease probs (same feat vector)
    dis_idx = int(np.argmax(d_probs))  # top-1 disease index within crop
    dis_name = d_classes_list[dis_idx]  # fine class folder name style
    dis_conf = float(d_probs[dis_idx]) * 100  # percentage confidence stage 2

    return crop_name, crop_conf, dis_name, dis_conf  # tuple for printing


crop_model_loaded = joblib.load("crop_model.pkl")  # reload from disk
with open("crop_classes.txt") as f:  # read label order
    crop_classes_loaded = [l.strip() for l in f if l.strip()]  # non-empty lines

disease_models_loaded, disease_classes_loaded = {}, {}  # dicts keyed by crop name
for c in CROPS:  # load each stage-2 artifact
    disease_models_loaded[c] = joblib.load(f"{c}_model.pkl")  # fitted RF
    with open(f"{c}_classes.txt") as f:  # matching class list
        disease_classes_loaded[c] = [l.strip() for l in f if l.strip()]  # strip whitespace

rand_cls = random.choice(os.listdir(DATASET_FLAT))  # random class folder name
rand_folder = os.path.join(DATASET_FLAT, rand_cls)  # path to that class
rand_img = random.choice(os.listdir(rand_folder))  # random file in folder
rand_path = os.path.join(rand_folder, rand_img)  # absolute image path

crop_name, crop_conf, dis_name, dis_conf = predict_two_stage_test(  # end-to-end smoke test
    rand_path, crop_model_loaded, crop_classes_loaded,
    disease_models_loaded, disease_classes_loaded
)

print(f"\nTest image : {rand_cls}/{rand_img}")  # picked sample
print(f"True class : {rand_cls}")  # folder name as ground-truth coarse+fine tag
print(f"Stage 1    : {crop_name}  ({crop_conf:.1f}%)")  # crop head output
print(f"Stage 2    : {dis_name}  ({dis_conf:.1f}%)")  # disease head output
print(f"Correct    : {'✓' if rand_cls == dis_name else '✗'}")  # stage-2 label vs truth
