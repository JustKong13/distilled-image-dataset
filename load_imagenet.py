import os
import numpy as np
from PIL import Image
from tqdm import tqdm

def load_tiny_imagenet(data_dir, image_size=(64, 64)):
    def load_images_from_folder(folder, label_map=None):
        images = []
        labels = []
        for label_idx, class_id in enumerate(os.listdir(folder)):
            class_folder = os.path.join(folder, class_id, "images")
            if not os.path.isdir(class_folder):
                continue
            for fname in os.listdir(class_folder):
                img_path = os.path.join(class_folder, fname)
                try:
                    img = Image.open(img_path).resize(image_size)
                    images.append(np.array(img))
                    labels.append(label_idx)
                except:
                    continue
        return np.array(images), np.array(labels)

    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val', 'images')

    # Load train data
    X_train, y_train = load_images_from_folder(train_dir)

    # Load val data
    val_annotations_file = os.path.join(data_dir, 'val', 'val_annotations.txt')
    val_annotations = {}
    with open(val_annotations_file) as f:
        for line in f:
            tokens = line.strip().split('\t')
            val_annotations[tokens[0]] = tokens[1]

    val_labels_map = {class_id: idx for idx, class_id in enumerate(sorted(os.listdir(train_dir)))}
    X_test = []
    y_test = []

    for fname in os.listdir(val_dir):
        class_id = val_annotations.get(fname)
        if class_id is None:
            continue
        label = val_labels_map[class_id]
        img_path = os.path.join(val_dir, fname)
        try:
            img = Image.open(img_path).convert("RGB").resize(image_size)
            X_test.append(np.array(img))
            y_test.append(label)
        except:
            continue

    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)