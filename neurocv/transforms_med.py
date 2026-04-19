"""BT kesitleri için kontrollü ön işleme ve augmentasyon (torchvision)."""
from __future__ import annotations

import random

import torchvision.transforms.functional as TF
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms import InterpolationMode

IMG_SIZE = 256


def train_cls_transforms(img_size: int = IMG_SIZE) -> T.Compose:
    return T.Compose(
        [
            T.Grayscale(num_output_channels=1),
            T.Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomAffine(degrees=8, translate=(0.02, 0.02), scale=(0.96, 1.04)),
            T.ToTensor(),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25]),
        ]
    )


def eval_cls_transforms(img_size: int = IMG_SIZE) -> T.Compose:
    return T.Compose(
        [
            T.Grayscale(num_output_channels=1),
            T.Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25]),
        ]
    )


def inference_cls_tensor(img: Image.Image, img_size: int = IMG_SIZE):
    """PIL RGB/L -> (1,3,H,W) normalize tensor."""
    img = img.convert("L")
    img = T.Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR)(img)
    t = T.ToTensor()(img).repeat(3, 1, 1)
    t = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])(t)
    return t.unsqueeze(0)


def preprocess_seg_pair_train(
    img: Image.Image,
    mask: Image.Image,
    img_size: int,
) -> tuple:
    """Aynı rastgele yatay çevirmeyi görüntü ve maskeye uygular."""
    img = img.convert("L")
    mask = mask.convert("L")
    img = TF.resize(img, (img_size, img_size), interpolation=InterpolationMode.BILINEAR)
    mask = TF.resize(mask, (img_size, img_size), interpolation=InterpolationMode.NEAREST)
    if random.random() < 0.5:
        img = TF.hflip(img)
        mask = TF.hflip(mask)
    img_t = TF.to_tensor(img).repeat(3, 1, 1)
    img_t = TF.normalize(img_t, mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
    mask_t = TF.to_tensor(mask)
    return img_t, mask_t


def preprocess_seg_pair_eval(
    img: Image.Image,
    mask: Image.Image,
    img_size: int,
) -> tuple:
    img = img.convert("L")
    mask = mask.convert("L")
    img = TF.resize(img, (img_size, img_size), interpolation=InterpolationMode.BILINEAR)
    mask = TF.resize(mask, (img_size, img_size), interpolation=InterpolationMode.NEAREST)
    img_t = TF.to_tensor(img).repeat(3, 1, 1)
    img_t = TF.normalize(img_t, mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
    mask_t = TF.to_tensor(mask)
    return img_t, mask_t


def inference_seg_tensors(pil_rgb: Image.Image, img_size: int):
    """Çıkarım: maske yokken sadece görüntü tensörü."""
    img = pil_rgb.convert("L")
    img = TF.resize(img, (img_size, img_size), interpolation=InterpolationMode.BILINEAR)
    img_t = TF.to_tensor(img).repeat(3, 1, 1)
    img_t = TF.normalize(img_t, mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
    return img_t.unsqueeze(0)
