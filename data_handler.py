import os
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image




# 2. Datasetin lataus ja esikäsittely

# 2.1 Datasetin jakaminen koulutus-, validointi- ja testijoukkoihin
def split_dataset(base_dir):
    images_dir = os.path.join(base_dir, 'Images')
    labels_dir = os.path.join(base_dir, 'Labels')

    image_files = sorted(os.listdir(images_dir))
    random.seed(42)
    random.shuffle(image_files)

    # Jaa dataset: 360 koulutukseen, 20 validointiin, 20 testiin
    train_files = image_files[:360]
    val_files = image_files[360:380]
    test_files = image_files[380:400]

    splits = {'train': train_files, 'val': val_files, 'test': test_files}

    for split in splits:
        split_images_dir = os.path.join(base_dir, split, 'images')
        split_labels_dir = os.path.join(base_dir, split, 'labels')
        os.makedirs(split_images_dir, exist_ok=True)
        os.makedirs(split_labels_dir, exist_ok=True)

        for file_name in splits[split]:
            # Kopioi kuva
            src_image = os.path.join(images_dir, file_name)
            dst_image = os.path.join(split_images_dir, file_name)
            if not os.path.exists(dst_image):
                shutil.copy(src_image, dst_image)

            # Luo vastaava label-tiedostonimi
            base_name = os.path.splitext(file_name)[0]
            label_file_name = f"{base_name}_mask.png"

            # Kopioi label
            src_label = os.path.join(labels_dir, label_file_name)
            dst_label = os.path.join(split_labels_dir, label_file_name)
            if not os.path.exists(dst_label):
                shutil.copy(src_label, dst_label)

# Suorita datasetin jakaminen
import shutil
if __name__ == "__main__":
    base_dir = 'Liver_Medical_Image_Datasets'
    split_dataset(base_dir)

# 2.2 Datasetin määrittely
class LiverDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform

        self.image_files = sorted(os.listdir(images_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        base_name = os.path.splitext(img_name)[0]
        label_name = f"{base_name}_mask.png"

        img_path = os.path.join(self.images_dir, img_name)
        label_path = os.path.join(self.labels_dir, label_name)

        image = Image.open(img_path).convert('RGB')
        label = Image.open(label_path).convert('L')

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)
            label = (label > 0).float()  # Binaarinen maski

        return image, label

# 2.3 Data transformaatiot
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# 2.4 DataLoaderit
def get_dataloaders(batch_size=4):
    base_dir = 'Liver_Medical_Image_Datasets'

    train_dataset = LiverDataset(
        images_dir=os.path.join(base_dir, 'train', 'images'),
        labels_dir=os.path.join(base_dir, 'train', 'labels'),
        transform=transform
    )

    val_dataset = LiverDataset(
        images_dir=os.path.join(base_dir, 'val', 'images'),
        labels_dir=os.path.join(base_dir, 'val', 'labels'),
        transform=transform
    )

    test_dataset = LiverDataset(
        images_dir=os.path.join(base_dir, 'test', 'images'),
        labels_dir=os.path.join(base_dir, 'test', 'labels'),
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return train_loader, val_loader, test_loader