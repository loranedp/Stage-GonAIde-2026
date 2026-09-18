import os
import json
import random
import argparse
from collections import defaultdict
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np

# Foreground category ids from _annotations.coco.json, in channel order (R, G, B).
# Category 0 is the Roboflow supercategory placeholder and has no annotations.
FOREGROUND_CATEGORY_IDS = [1, 2, 3]  # Cavite, Gonade, Intestin


class CocoMultiLabelDataset(Dataset):
    """Multi-label segmentation dataset backed by a COCO polygon annotation file.

    Each foreground category is rasterized into its own channel, so overlapping
    polygons of different classes coexist (one pixel can carry several labels).
    The mask is kept as an RGB PIL image (one class per channel, 255 = present)
    so the paired PIL transforms in dataset/transforms.py apply unchanged.
    Returns (image, mask, file_name) like TN3KDataset.
    """

    def __init__(self, coco_json, images_dir, split_file, split='train', transform=None):
        self.images_dir = images_dir
        self.transform = transform

        with open(split_file, 'r') as f:
            splits = json.load(f)
        if split not in splits:
            raise ValueError(f"Split must be one of {list(splits.keys())}, got {split}")
        wanted = set(splits[split])

        with open(coco_json, 'r') as f:
            coco = json.load(f)

        anns_by_image = defaultdict(list)
        for ann in coco['annotations']:
            anns_by_image[ann['image_id']].append(ann)

        self.samples = []
        for img in coco['images']:
            if img['file_name'] in wanted:
                self.samples.append({
                    'file_name': img['file_name'],
                    'width': img['width'],
                    'height': img['height'],
                    'annotations': anns_by_image[img['id']],
                })
        self.samples.sort(key=lambda s: s['file_name'])

        found = {s['file_name'] for s in self.samples}
        missing = wanted - found
        if missing:
            raise ValueError(f"{len(missing)} file(s) in split '{split}' not found in {coco_json}: "
                             f"{sorted(missing)[:3]}...")

    def __len__(self):
        return len(self.samples)

    def _build_mask(self, sample):
        size = (sample['width'], sample['height'])
        layers = {cid: Image.new('L', size, 0) for cid in FOREGROUND_CATEGORY_IDS}
        draws = {cid: ImageDraw.Draw(layers[cid]) for cid in FOREGROUND_CATEGORY_IDS}
        for ann in sample['annotations']:
            cid = ann['category_id']
            if cid not in layers:
                continue
            for poly in ann['segmentation']:
                if len(poly) < 6:  # need at least 3 points
                    continue
                points = list(zip(poly[0::2], poly[1::2]))
                draws[cid].polygon(points, fill=255)
        return Image.merge('RGB', [layers[cid] for cid in FOREGROUND_CATEGORY_IDS])

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = os.path.join(self.images_dir, sample['file_name'])
        image = Image.open(img_path).convert('RGB')
        mask = self._build_mask(sample)

        if self.transform:
            image, mask = self.transform(image, mask)
        else:
            image = transforms.ToTensor()(image)
            mask = (torch.from_numpy(np.array(mask)).permute(2, 0, 1) > 127).float()

        return image, mask, sample['file_name']


def make_splits(coco_json, out, seed=42, ratios=(0.7, 0.15, 0.15)):
    with open(coco_json, 'r') as f:
        coco = json.load(f)
    file_names = sorted(img['file_name'] for img in coco['images'])
    rng = random.Random(seed)
    rng.shuffle(file_names)

    n = len(file_names)
    n_train = round(n * ratios[0])
    n_val = round(n * ratios[1])
    splits = {
        'train': sorted(file_names[:n_train]),
        'val': sorted(file_names[n_train:n_train + n_val]),
        'test': sorted(file_names[n_train + n_val:]),
    }
    with open(out, 'w') as f:
        json.dump(splits, f, indent=2)
    print(f"Wrote {out}: {len(splits['train'])} train / {len(splits['val'])} val / "
          f"{len(splits['test'])} test (seed {seed})")
    return splits


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Build train/val/test splits for a COCO segmentation dataset')
    parser.add_argument('--make_splits', action='store_true')
    parser.add_argument('--coco_json', required=True, type=str)
    parser.add_argument('--out', required=True, type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--ratios', nargs=3, default=[0.7, 0.15, 0.15], type=float)
    args = parser.parse_args()
    make_splits(args.coco_json, args.out, seed=args.seed, ratios=tuple(args.ratios))
