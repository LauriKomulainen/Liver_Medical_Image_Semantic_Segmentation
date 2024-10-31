import os
import torch
from PIL import Image

from data_handler import get_dataloaders, transform
from model import UNet, train_model, compute_metrics, segment_single_image
import matplotlib.pyplot as plt
from datetime import datetime

# Hyperparametrit
batch_size = 4
num_epochs = 1
learning_rate = 1e-4

if __name__ == "__main__":
    result_dir = "model reports"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(result_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    try:
        train_loader, val_loader, test_loader = get_dataloaders(batch_size=batch_size)
        print("DataLoaderit ladattu")

        # Mallin lataus tai koulutus
        model = UNet()
        model_path = 'unet_model.pth'
        if os.path.exists(model_path):
            print("Malli löytyy tiedostosta, ladataan se...")
            model.load_state_dict(torch.load(model_path))
            train_losses, val_losses = None, None
        else:
            print("Malli ei löytynyt tiedostosta, aloitetaan koulutus...")
            train_losses, val_losses = train_model(model, train_loader, val_loader)
            torch.save(model.state_dict(), model_path)
            print("Koulutus suoritettu ja malli tallennettu")

    except Exception as e:
        print(f"Virhe suorituksen aikana: {e}")

    loss_plot_path = None

    existing_loss_plot_path = os.path.abspath('loss_progression.png')
    if os.path.exists(existing_loss_plot_path):
        loss_plot_path = existing_loss_plot_path
    else:
        if train_losses is not None and val_losses is not None:
            plt.figure()
            plt.plot(train_losses, label='Train Loss')
            plt.plot(val_losses, label='Validation Loss')
            plt.title('Loss Progression')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            loss_plot_path = os.path.join(output_dir, 'loss_progression.png')
            plt.savefig(loss_plot_path)
            plt.show()
        else:
            loss_plot_path = "Ei koulutustietoa saatavilla"

    mIoU, mPA = compute_metrics(model, test_loader)

    def arvioi_miou(mIoU):
        if mIoU > 0.85:
            return (
                f"Mean Intersection over Union (mIoU) on erinomainen ({mIoU:.4f}), mikä osoittaa, että malli suoriutuu erittäin hyvin "
                "maksakuvien segmentoinnista ja pystyy erottamaan maksakudoksen tarkasti taustasta ja muista kudoksista."
            )
        elif 0.75 <= mIoU <= 0.85:
            return (
                f"Mean Intersection over Union (mIoU) on hyvä ({mIoU:.4f}), mikä osoittaa, että malli suoriutuu segmentoinnista "
                "luotettavasti. Vaikka tarkkuus on korkea, pieniä parannuksia voi olla tarpeen joillakin alueilla."
            )
        elif 0.5 <= mIoU < 0.75:
            return (
                f"Mean Intersection over Union (mIoU) on hyväksyttävä ({mIoU:.4f}), mikä viittaa siihen, että malli suoriutuu segmentoinnista "
                "tyydyttävästi, mutta tarkkuuden parantaminen on suositeltavaa tehokkaan kliinisen sovelluksen varmistamiseksi."
            )
        else:
            return (
                f"Mean Intersection over Union (mIoU) on matala ({mIoU:.4f}), mikä osoittaa, että mallin segmentointitarkkuus on riittämätön "
                "ja vaatii huomattavia parannuksia."
            )

    def arvioi_mpa(mPA):
        if mPA > 0.9:
            return (
                f"Mean Pixel Accuracy (mPA) on erittäin korkea ({mPA:.4f}), mikä tarkoittaa, että malli osaa luokitella pikselit erittäin tarkasti "
                "maksakuvissa, mikä tekee siitä hyvin soveltuvan kliinisiin sovelluksiin."
            )
        elif 0.85 <= mPA <= 0.9:
            return (
                f"Mean Pixel Accuracy (mPA) on korkea ({mPA:.4f}), mikä osoittaa, että malli pystyy luotettavasti luokittelemaan suurimman osan pikseleistä "
                "oikein. Pieniä parannuksia voi harkita joidenkin alueiden tarkkuuden nostamiseksi."
            )
        elif 0.7 <= mPA < 0.85:
            return (
                f"Mean Pixel Accuracy (mPA) on hyväksyttävä ({mPA:.4f}), mikä tarkoittaa, että malli suoriutuu segmentoinnista pikselitasolla tyydyttävästi, "
                "mutta tarkkuutta olisi hyvä parantaa laadun varmistamiseksi."
            )
        else:
            return (
                f"Mean Pixel Accuracy (mPA) on matala ({mPA:.4f}), mikä viittaa siihen, että malli ei onnistu luotettavasti pikselien oikeassa luokittelussa, "
                "ja sen käyttö kliinisissä sovelluksissa on rajoitettua."
            )

    report_path = os.path.join(output_dir, 'arviointiraportti.html')
    with open(report_path, 'w') as f:
        # Kirjoita HTML:n alkuosa
        f.write("<html><head><title>Arviointiraportti</title></head><body>")
        f.write("<h1>Mallin arviointi testijoukolla</h1>")
        f.write(f"<p>Keskimääräinen IoU: {mIoU:.4f}</p>")
        f.write(f"<p>Keskimääräinen pikselitarkkuus (mPA): {mPA:.4f}</p>")
        f.write("<h2>Analyysi:</h2>")
        f.write(f"<p>{arvioi_miou(mIoU)}</p>")
        f.write(f"<p>{arvioi_mpa(mPA)}</p>")
        f.write(f"<p>Koulutuskierrokset: {num_epochs}, Oppimisnopeus: {learning_rate}, Eräkoko: {batch_size}</p>")
        f.write("<h2>Luodut kuvaajat ja kuvat:</h2>")

        if os.path.exists(loss_plot_path):
            f.write("<h3>Häviön kehityskaavio:</h3>")
            f.write(f"<img src='{loss_plot_path}' width='800'><br>")
        else:
            f.write("<p>Häviön kehityskaavio: Ei koulutustietoa saatavilla</p>")

        test_images_dir = 'Liver_Medical_Image_Datasets/test/images'
        test_image_files = sorted(os.listdir(test_images_dir))

        test_images = test_image_files[:3]
        for img_name in test_images:
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
            output_image_path = os.path.abspath(os.path.join(output_dir, f'segmentation_result_{base_name}.png'))
            plt.savefig(output_image_path)
            plt.close(fig)

            thumbnail_path = os.path.abspath(os.path.join(output_dir, f'segmentation_result_{base_name}_thumbnail.png'))
            image_obj = Image.open(output_image_path)
            image_obj.thumbnail((800, 800))
            image_obj.save(thumbnail_path)

            f.write(f"<h3>Segmentointikuva: {base_name}</h3>")
            f.write(f"<a href='{output_image_path}' target='_blank'><img src='{thumbnail_path}' width='800'></a><br>")

        f.write("</body></html>")