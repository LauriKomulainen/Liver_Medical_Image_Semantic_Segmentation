import os
import random
import shutil

def split_dataset(base_dir, train_ratio=0.9, val_ratio=0.1):
    images_dir = os.path.join(base_dir, 'Images')
    labels_dir = os.path.join(base_dir, 'Labels')

    image_files = os.listdir(images_dir)
    image_files.sort()
    random.seed(42)
    random.shuffle(image_files)

    total_images = len(image_files)
    train_end = int(total_images * train_ratio)
    val_end = train_end + int(total_images * val_ratio)

    splits = {
        'train': image_files[:train_end],
        'val': image_files[train_end:val_end],
        'test': image_files[val_end:]
    }

    for split in splits:
        split_images_dir = os.path.join(base_dir, split, 'images')
        split_labels_dir = os.path.join(base_dir, split, 'labels')
        os.makedirs(split_images_dir, exist_ok=True)
        os.makedirs(split_labels_dir, exist_ok=True)

        for file_name in splits[split]:
            # Kopioi kuva
            shutil.copy(os.path.join(images_dir, file_name), split_images_dir)

            # Luo vastaava label-tiedostonimi
            base_name = os.path.splitext(file_name)[0]
            label_file_name = f"{base_name}_mask.png"

            # Kopioi label
            shutil.copy(os.path.join(labels_dir, label_file_name), split_labels_dir)

if __name__ == "__main__":
    base_dir = 'Liver_Medical_Image_Datasets'
    split_dataset(base_dir)
