from PIL import Image
import torchvision.transforms.functional as F
from torchvision import transforms


# Keep the custom OpenUS pipeline aligned with the augmentations used by UNet
# and YOLO. Geometry is applied to image and mask with the same parameters;
# only the image receives the photometric transform.
ROTATION_DEGREES = (-20.0, 20.0)
TRANSLATION_RATIO = (0.2, 0.2)
SCALE_RANGE = (0.8, 1.2)
BRIGHTNESS = 0.4


def get_transforms(img_size):
    # Single-label masks: [H, W] long labels (binary 0/255 pngs become 0/1)
    mask_to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.squeeze(0).long())  # Remove channel dim and convert to long
    ])
    return _get_transforms(img_size, mask_to_tensor)


def get_transforms_multilabel(img_size):
    # Multi-label masks: RGB PIL image with one class per channel (255 = present)
    # becomes a [C, H, W] float tensor of {0, 1}, as expected by BCEWithLogitsLoss
    mask_to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x > 0.5).float())
    ])
    return _get_transforms(img_size, mask_to_tensor)


def _get_transforms(img_size, mask_to_tensor):
    def resize(im, msk):
        im = F.resize(im, [img_size, img_size], interpolation=transforms.InterpolationMode.BILINEAR)
        msk = F.resize(msk, [img_size, img_size], interpolation=Image.NEAREST)
        return im, msk

    def random_geometry(im, msk):
        rotation = transforms.RandomRotation.get_params(ROTATION_DEGREES)
        affine_rotation, translations, scale, shear = transforms.RandomAffine.get_params(
            degrees=[0.0, 0.0],
            translate=list(TRANSLATION_RATIO),
            scale_ranges=SCALE_RANGE,
            shears=None,
            img_size=[img_size, img_size],
        )

        im = F.rotate(
            im, rotation,
            interpolation=transforms.InterpolationMode.BILINEAR,
            fill=0,
        )
        msk = F.rotate(
            msk, rotation,
            interpolation=Image.NEAREST,
            fill=0,
        )
        im = F.affine(
            im, affine_rotation, translations, scale, shear,
            interpolation=transforms.InterpolationMode.BILINEAR,
            fill=0,
        )
        msk = F.affine(
            msk, affine_rotation, translations, scale, shear,
            interpolation=Image.NEAREST,
            fill=0,
        )
        return im, msk

    # ToTensor and Normalize only apply to image
    image_to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
    ])

    def train_transform(image, mask):
        image, mask = resize(image, mask)
        image, mask = random_geometry(image, mask)
        image = transforms.ColorJitter(brightness=BRIGHTNESS)(image)
        image = image_to_tensor(image)
        mask = mask_to_tensor(mask)
        return image, mask

    def val_transform(image, mask):
        image, mask = resize(image, mask)
        image = image_to_tensor(image)
        mask = mask_to_tensor(mask)
        return image, mask

    return train_transform, val_transform
