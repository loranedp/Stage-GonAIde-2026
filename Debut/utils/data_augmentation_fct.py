import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

#Fonction pour la Data augmentation
def data_augmentation(image, mask):

    # On récupère le nombre de canaux de l'image et du masque
    img_channels = image.shape[0]
    mask_channels = mask.shape[0]

    # 1. Transformation de base : redimensionnement
    img_resize = T.Resize((512, 512), antialias=True)
    mask_resize = T.Resize((512, 512), interpolation=T.InterpolationMode.NEAREST)
    image = img_resize(image)
    mask = mask_resize(mask)

    # 2. Concaténation de l'image et du masque pour appliquer les mêmes transformations
    combined = torch.cat((image, mask), dim=0)

    # 3. Transformations géométriques : rotation, mise à l'échelle, translation
    geom_transforms = T.Compose([
        T.RandomRotation(degrees=20),
        T.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.8, 1.2))
    ])
    combined = geom_transforms(combined)
    image = combined[0:img_channels, :, :]
    mask = combined[img_channels:img_channels+mask_channels, :, :]

    # 4. Transformations de couleur : ajustement de la luminosité
    bright=T.ColorJitter(brightness=0.4)
    image_rgb = image[0:3, :, :] # Les 3 canaux RGB d'origine
    image_meta = image[3:, :, :]

    image_rgb = bright(image_rgb)
    image = torch.cat((image_rgb, image_meta), dim=0)

    return image, mask


# Fonction pour la Data augmentation d'une séquence de coupes (dépendance inter-coupes) :
# applique la MEME transformation (géométrique et de luminosité) à toutes les coupes,
# pour préserver la correspondance spatiale entre coupes exploitée par
# CrossSliceAttention / xLSTMSliceMixer (voir model_resnet.ipynb).
def data_augmentation_sequence(images, masks):
    """
    Args:
        images (torch.Tensor): (T, C, H, W) - coupes déjà à la taille cible.
        masks (torch.Tensor): (T, num_classes, H, W)
    """
    n_slices, img_channels, H, W = images.shape
    _, mask_channels, _, _ = masks.shape

    images_flat = images.reshape(n_slices * img_channels, H, W)
    masks_flat = masks.reshape(n_slices * mask_channels, H, W)

    # Concaténation image+masque de TOUTES les coupes : une seule transformation
    # géométrique tirée pour l'ensemble de la séquence
    combined = torch.cat((images_flat, masks_flat), dim=0)
    geom_transforms = T.Compose([
        T.RandomRotation(degrees=20),
        T.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.8, 1.2))
    ])
    combined = geom_transforms(combined)
    images_flat = combined[0:n_slices * img_channels]
    masks_flat = combined[n_slices * img_channels:]

    # Luminosité : un seul facteur tiré pour toute la séquence, appliqué à toutes les
    # coupes. adjust_brightness n'accepte que des tenseurs à 1 ou 3 canaux (contrairement
    # à RandomRotation/RandomAffine ci-dessus) : on ne peut donc pas empiler les coupes
    # sur l'axe des canaux comme pour la transformation géométrique. On passe un batch
    # (T, 3, H, W) à adjust_brightness, qui applique le même facteur à chaque coupe.
    images_seq = images_flat.view(n_slices, img_channels, H, W)
    images_rgb = images_seq[:, 0:3]  # Les 3 canaux RGB d'origine de chaque coupe
    images_meta = images_seq[:, 3:]  # Canaux méta CME éventuels, non affectés par le jitter

    _, brightness_factor, _, _, _ = T.ColorJitter.get_params([0.6, 1.4], None, None, None)
    images_rgb = TF.adjust_brightness(images_rgb, brightness_factor)

    images = torch.cat((images_rgb, images_meta), dim=1)
    masks = masks_flat.view(n_slices, mask_channels, H, W)
    return images, masks