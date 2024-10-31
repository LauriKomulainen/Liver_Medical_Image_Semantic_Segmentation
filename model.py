import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import os
from config import learning_rate, num_epochs, batch_size


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # Määritellään U-Netin peruspalikat
        def conv_block(in_channels, out_channels):
            block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            return block

        def up_conv(in_channels, out_channels):
            return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        # U-Netin kerrokset
        self.conv1 = conv_block(3, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = conv_block(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = conv_block(512, 1024)

        self.up6 = up_conv(1024, 512)
        self.conv6 = conv_block(1024, 512)

        self.up7 = up_conv(512, 256)
        self.conv7 = conv_block(512, 256)

        self.up8 = up_conv(256, 128)
        self.conv8 = conv_block(256, 128)

        self.up9 = up_conv(128, 64)
        self.conv9 = conv_block(128, 64)

        self.conv10 = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        c1 = self.conv1(x)
        p1 = self.pool1(c1)

        c2 = self.conv2(p1)
        p2 = self.pool2(c2)

        c3 = self.conv3(p2)
        p3 = self.pool3(c3)

        c4 = self.conv4(p3)
        p4 = self.pool4(c4)

        # Bottleneck
        c5 = self.conv5(p4)

        # Decoder
        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6 = self.conv6(merge6)

        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)

        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)

        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)

        c10 = self.conv10(c9)
        return c10


def train_model(model, train_loader, val_loader, num_epochs, learning_rate):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []

    print("Koulutus alkaa...")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        print(f"Epoch {epoch + 1}/{num_epochs}")

        for batch_idx, (images, masks) in enumerate(train_loader):
            print(f"  Erä {batch_idx + 1}/{len(train_loader)} käsittelyssä...")
            print(f"  Kuvan muoto: {images.shape}, Maskin muoto: {masks.shape}")

            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validointi
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    # Tallenna malli
    torch.save(model.state_dict(), 'unet_model.pth')
    print("Koulutus valmis ja malli tallennettu.")

    return train_losses, val_losses


def segment_single_image(model, image_path, transform):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    image = Image.open(image_path).convert('RGB')
    image_transformed = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_transformed)
        output = torch.sigmoid(output)
        output = (output > 0.5).float()

    output_np = output.cpu().squeeze().numpy()

    # Ladataan ground truth
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    label_name = f"{base_name}_mask.png"
    label_path = os.path.join(os.path.dirname(image_path).replace('images', 'labels'), label_name)

    print("Image Path:", image_path)
    print("Base Name:", base_name)
    print("Label Path:", label_path)
    label = Image.open(label_path).convert('L')
    label = transform(label).squeeze().numpy()



    return image, label, output_np

def compute_metrics(model, data_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    total_iou = 0.0
    total_acc = 0.0
    n_samples = 0

    with torch.no_grad():
        for images, masks in data_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            outputs = torch.sigmoid(outputs)
            preds = (outputs > 0.5).float()

            intersection = (preds * masks).sum(dim=(2, 3))
            union = preds.sum(dim=(2, 3)) + masks.sum(dim=(2, 3)) - intersection
            iou = (intersection + 1e-6) / (union + 1e-6)
            acc = (preds == masks).float().mean(dim=(2, 3))

            total_iou += iou.sum().item()
            total_acc += acc.sum().item()
            n_samples += images.size(0)

    mIoU = total_iou / n_samples
    mPA = total_acc / n_samples
    return mIoU, mPA
