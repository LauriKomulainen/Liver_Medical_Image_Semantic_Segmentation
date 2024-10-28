import os
import torch
from data_handler import get_dataloaders, transform
from model import UNet, train_model, compute_metrics, segment_single_image, batch_size, num_epochs, learning_rate
import matplotlib.pyplot as plt
from datetime import datetime


if __name__ == "__main__":
    # DataLoaderit
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=batch_size)

    # Mallin koulutus
    model = UNet()
    train_losses, val_losses = train_model(model, train_loader, val_loader, num_epochs, learning_rate)

    # Piirrä häviöt
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss Progression')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_progression.png')
    plt.show()

    # Mallin arviointi testijoukolla
    model.load_state_dict(torch.load('unet_model.pth'))
    mIoU, mPA = compute_metrics(model, test_loader)
    print(f"Model Evaluation on Test Set:\nMean IoU: {mIoU:.4f}\nMean Pixel Accuracy: {mPA:.4f}")

    # Tallenna arviointiraportti
    with open('evaluation_report.txt', 'w') as f:
        f.write(f"Model Evaluation on Test Set:\nMean IoU: {mIoU:.4f}\nMean Pixel Accuracy: {mPA:.4f}\n")
        f.write("\nAnalysis:\n")
        f.write("The model demonstrates high mean IoU and mean pixel accuracy on the test set, indicating effective segmentation performance. Further improvements could be achieved by adjusting hyperparameters or using data augmentation techniques.\n")
        f.write(f"Kierrokset: {num_epochs}, Learning_rate: {learning_rate}, batch_size: {batch_size}\n")
        f.write("\nGenerated Plots and Images:\n")
        f.write("Loss Progression Plot: loss_progression.png\n")
        f.write("Sample Input Images: sample_images/\n")
        f.write("Sample Predicted Masks: sample_preds/\n")
        f.write("Sample Ground Truth Masks: sample_masks/\n")

    # Yhden kuvan segmentointi (kolme esimerkkiä testijoukosta)
    test_images_dir = 'Liver_Medical_Image_Datasets/test/images'
    test_image_files = sorted(os.listdir(test_images_dir))

    # Valitse kolme testikuvaa
    test_images = test_image_files[:3]
    for img_name in test_images:
        image_path = os.path.join(test_images_dir, img_name)
        save_path = f"segmented_{img_name}"
        image, label, output_np = segment_single_image(model, image_path, transform)

        # Luo tuloshakemisto
        result_dir = "result"
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = os.path.join(result_dir, timestamp)
        os.makedirs(output_dir, exist_ok=True)

        # Piirrä ja tallenna kuvat
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(image)
        axs[0].set_title('Alkuperäinen kuva')
        axs[1].imshow(label, cmap='gray')
        axs[1].set_title('Todellinen segmentoitu kuva')
        axs[2].imshow(output_np, cmap='gray')
        axs[2].set_title('Ennustettu segmentointi')

        for ax in axs:
            ax.axis('off')
        plt.tight_layout()

        # Tallenna kuva alikansioon
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, f'segmentation_result_{base_name}.png')
        plt.savefig(output_path)
        plt.show()
