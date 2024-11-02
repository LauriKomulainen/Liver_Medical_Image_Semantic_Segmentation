import os
import random
import torch
from PIL import Image
from data_handler import get_dataloaders, transform
from model import UNet, train_model, compute_metrics, segment_single_image, random_search
import matplotlib.pyplot as plt
from datetime import datetime

if __name__ == "__main__":
    result_dir = "model reports"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_output_dir = os.path.join(result_dir, timestamp)
    os.makedirs(base_output_dir, exist_ok=True)

    param_space = {
        'batch_size': [4, 8, 16],
        'learning_rate': [1e-3, 1e-4, 1e-5],
        'dropout_rate': [0.1, 0.2, 0.3, 0.4, 0,5],
        'optimizer_type': ['Adam', 'SGD'],
        'num_epochs': [5, 10, 15, 20, 30]
    }

    train_loader, val_loader, test_loader = get_dataloaders(batch_size=4)
    print("Data ladattu")

    num_trials = 2
    print("Suoritetaan satunnainen hyperparametrien valinta...")

    for trial in range(num_trials):
        params = {
            'batch_size': random.choice(param_space['batch_size']),
            'learning_rate': random.choice(param_space['learning_rate']),
            'dropout_rate': random.choice(param_space['dropout_rate']),
            'optimizer_type': random.choice(param_space['optimizer_type']),
            'num_epochs': random.choice(param_space['num_epochs'])
        }

        model = UNet(dropout_rate=params['dropout_rate'])
        output_dir = os.path.join(base_output_dir, f"kierros_{trial + 1}")
        os.makedirs(output_dir, exist_ok=True)

        print(f"\nKierros (Epoch): {trial + 1}/{num_trials} - Hyperparametrit: {params}")

        train_losses, val_losses = train_model(
            model, train_loader, val_loader,
            optimizer_type=params['optimizer_type'],
            learning_rate=params['learning_rate'],
            num_epochs=params['num_epochs'],
            dropout_rate=params['dropout_rate'],
            early_stopping_patience=5
        )

        # Tallenna koulutettu malli kokeilukansioon
        model_path = os.path.join(output_dir, 'unet_model.pth')
        torch.save(model.state_dict(), model_path)

        # Häviön kuvaajan luonti
        if train_losses and val_losses:
            plt.figure()
            plt.plot(train_losses, label='Harjoitushäviö')
            plt.plot(val_losses, label='Validointihäviö')
            plt.title('Häviön kehitys')
            plt.xlabel('Kierrokset (Epochs)')
            plt.ylabel('Häviö (Loss)')
            plt.legend()
            loss_plot_path = os.path.join(output_dir, 'loss_progression.png')
            plt.savefig(loss_plot_path)
            plt.close()
        else:
            loss_plot_path = "Ei häviötietoja saatavilla"

        # Mallin suorituskyvyn arviointi
        mIoU, mPA = compute_metrics(model, test_loader)
        print(f"Kierros {trial + 1} - mIoU: {mIoU}, mPA: {mPA}")

        # Luo raportti HTML-muodossa
        report_path = os.path.join(output_dir, 'arviointiraportti.html')
        with open(report_path, 'w') as f:
            f.write("<html><head><title>Arviointiraportti</title></head><body>")
            f.write(f"<h1>Malli {trial + 1}</h1>")
            f.write("<h2>mIoU & mPA:</h2>")
            f.write(f"<p>Mallin mIoU: {mIoU:.4f}</p>")
            f.write(f"<p>Mallin pikselitarkkuus (mPA): {mPA:.4f}</p>")

            f.write("<h2>Hyperparametrit:</h2>")
            f.write("<ul>")
            f.write(f"<li>Koulutuskierrokset (Epochs): {params['num_epochs']}</li>")
            f.write(f"<li>Oppimisnopeus (Learning Rate): {params['learning_rate']}</li>")
            f.write(f"<li>Eräkoko (Batch Size): {params['batch_size']}</li>")
            f.write(f"<li>Dropout Rate: {params['dropout_rate']}</li>")
            f.write(f"<li>Optimointialgoritmi (Optimizer): {params['optimizer_type']}</li>")
            f.write("</ul>")

            f.write("<h2>Kuvat:</h2>")

            if os.path.exists(loss_plot_path):
                f.write("<h3>Häviön kehityskaavio:</h3>")
                f.write(f"<img src='{os.path.basename(loss_plot_path)}' width='800'><br>")
            else:
                f.write("<p>Häviön kehityskaavio: Ei koulutustietoa saatavilla</p>")

            test_images_dir = 'Liver_Medical_Image_Datasets/test/images'
            test_image_files = sorted(os.listdir(test_images_dir))[:3]

            for img_name in test_image_files:
                image_path = os.path.join(test_images_dir, img_name)
                image, label, output_np = segment_single_image(model, image_path, transform)

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

                base_name = os.path.splitext(os.path.basename(image_path))[0]
                output_image_path = os.path.join(output_dir, f'segmentation_result_{base_name}.png')
                plt.savefig(output_image_path)
                plt.close(fig)

                thumbnail_path = os.path.join(output_dir, f'segmentation_result_{base_name}_thumbnail.png')
                image_obj = Image.open(output_image_path)
                image_obj.thumbnail((800, 800))
                image_obj.save(thumbnail_path)

                output_image_rel_path = os.path.basename(output_image_path)
                thumbnail_rel_path = os.path.basename(thumbnail_path)

                f.write(f"<h3>Segmentointikuva: {base_name}</h3>")
                f.write(
                    f"<a href='{output_image_rel_path}' target='_blank'><img src='{thumbnail_rel_path}' width='800'></a><br>")

            f.write("</body></html>")