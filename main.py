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

    # Tallenna arviointiraportti
    with open('arviointiraportti.txt', 'w') as f:
        f.write(f"Mallin arviointi testijoukolla:\nKeskimääräinen IoU: {mIoU:.4f}\nKeskimääräinen pikselitarkkuus (mPA): {mPA:.4f}\n")
        f.write("\nAnalyysi:\n")
        f.write(arvioi_miou(mIoU) + "\n")
        f.write(arvioi_mpa(mPA) + "\n")
        f.write(f"Koulutuskierrokset: {num_epochs}, Oppimisnopeus: {learning_rate}, Eräkoko: {batch_size}\n")
        f.write("\nLuodut kuvaajat ja kuvat:\n")
        f.write("Häviön kehityskaavio: loss_progression.png\n")
        f.write("Esimerkkisyötekuvat: sample_images/\n")
        f.write("Esimerkit ennustetuista segmenteista: sample_preds/\n")
        f.write("Esimerkit todellisista segmenteista: sample_masks/\n")

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
