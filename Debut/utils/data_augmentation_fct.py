import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

# Fonction pour appliquer les transformations géométriques
def apply_geometry(tensor, rotation, affine_rotation, translations, scale, shear, interpolation=T.InterpolationMode.BILINEAR):
    """Applique les mêmes paramètres géométriques à un tenseur."""
    tensor = TF.rotate(
        tensor, rotation, interpolation=interpolation, fill=0
    )
    return TF.affine(
        tensor,
        angle=affine_rotation,
        translate=translations,
        scale=scale,
        shear=shear,
        interpolation=interpolation,
        fill=0,
    )

# Fonction pour la Data augmentation
def data_augmentation(image, mask, model="classique"):

    # --- 1. Vérification des paramètres ---
    if model not in {"classique", "egg_hv"}:
        raise ValueError(f"Modèle d'augmentation inconnu : {model}")
    if model == "egg_hv" and mask.shape[0] != 3:
        raise ValueError("Le mode egg_hv attend les cibles [masque, H, V]")

    # --- 2. Tirage des paramètres de transformation géométrique ---
    rotation = T.RandomRotation.get_params([-20.0, 20.0])
    affine_rotation, translations, scale, shear = T.RandomAffine.get_params(
        degrees=[0.0, 0.0],
        translate=[0.1, 0.1],
        scale_ranges=[0.8, 1.2],
        shears=None,
        img_size=[image.shape[-2], image.shape[-1]],
    )

    # --- 3. Application des transformations : rotation, mise à l'échelle, translation ---
    image = apply_geometry(image, rotation, affine_rotation, translations, scale, shear, T.InterpolationMode.BILINEAR)

    if model == "egg_hv": # Différence car le masque contient un vecteur H/V orienté vers le centre de l'œuf
        egg_mask = apply_geometry(mask[0:1], rotation, affine_rotation, translations, scale, shear, T.InterpolationMode.NEAREST)
        hv = apply_geometry(mask[1:3], rotation, affine_rotation, translations, scale, shear, T.InterpolationMode.BILINEAR)

        # La rotation doit être appliquée au vecteur H/V pour que la direction reste correcte après rotation de l'image.
        theta = torch.deg2rad(torch.tensor(rotation, dtype=hv.dtype, device=hv.device))
        cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
        horizontal, vertical = hv[0:1], hv[1:2]
        hv = torch.cat(
            [
                cos_theta * horizontal - sin_theta * vertical,
                sin_theta * horizontal + cos_theta * vertical,
            ],
            dim=0,
        )
        egg_mask.clamp_(0, 1)
        hv.clamp_(-1, 1)
        mask = torch.cat((egg_mask, hv), dim=0)

    else:
        mask = apply_geometry(mask, rotation, affine_rotation, translations, scale, shear, T.InterpolationMode.NEAREST)

    # --- 4. Transformations de couleur : ajustement de la luminosité ---
    image_rgb = T.ColorJitter(brightness=0.4)(image[0:3])
    image = torch.cat((image_rgb, image[3:]), dim=0)

    return image, mask


# Fonction pour la Data augmentation d'une séquence de coupes (dépendance inter-coupes) pour appliquer les mêmes transformation à toutes les coupes.
def data_augmentation_sequence(images, masks):
    """Applique la même transformation à toutes les coupes d'une séquence."""
    # --- 1. Tirage unique des paramètres géométriques de la séquence ---
    rotation = T.RandomRotation.get_params([-20.0, 20.0])

    affine_rotation, translations, scale, shear = T.RandomAffine.get_params(
        degrees=[0.0, 0.0],
        translate=[0.1, 0.1],
        scale_ranges=[0.8, 1.2],
        shears=None,
        img_size=[images.shape[-2], images.shape[-1]],
    )

    # --- 2. Transformation géométrique des images ---
    images = apply_geometry(images, rotation, affine_rotation, translations, scale, shear, T.InterpolationMode.BILINEAR)

    # --- 3. Transformation géométrique des masques ---
    masks = apply_geometry(masks, rotation, affine_rotation, translations, scale, shear, T.InterpolationMode.NEAREST)

    # --- 4. Transformation de luminosité des images ---
    images_rgb = images[:, 0:3]  # Les 3 canaux RGB d'origine de chaque coupe
    images_meta = images[:, 3:]  # Canaux méta CME éventuels

    _, brightness_factor, _, _, _ = T.ColorJitter.get_params([0.6, 1.4], None, None, None)
    images_rgb = TF.adjust_brightness(images_rgb, brightness_factor)

    images = torch.cat((images_rgb, images_meta), dim=1)

    return images, masks
