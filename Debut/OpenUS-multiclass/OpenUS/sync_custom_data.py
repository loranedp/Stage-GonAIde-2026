"""Synchronize local OpenUS data with Debut/data and the common test fish."""

import argparse
import json
from pathlib import Path
import random
import shutil
import tempfile


def prepare(debut_root):
    source = debut_root / 'data/images/cavite'
    coco = json.loads((debut_root / 'data/labels/COCO/cavite/annotations.json').read_text())
    test_fish = set(json.loads((debut_root / 'utils/common_test_fish.json').read_text()))
    categories = {c['id']: c['name'] for c in coco['categories']}
    if any(categories.get(cid) != name for cid, name in
           [(1, 'Cavite'), (2, 'Gonade'), (3, 'Intestin')]):
        raise ValueError('Expected category IDs 1=Cavite, 2=Gonade, 3=Intestin')
    names = [image['file_name'] for image in coco['images']]
    ids = [image['id'] for image in coco['images']]
    if len(set(names)) != len(names) or len(set(ids)) != len(ids):
        raise ValueError('Duplicate image names or IDs')
    fish = {}
    for name in names:
        if Path(name).name != name or len(name.split('_')) < 4:
            raise ValueError(f'Invalid image name: {name}')
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)
        fish[name] = name.split('_')[2]
    image_ids = set(ids)
    for annotation in coco['annotations']:
        if annotation['image_id'] not in image_ids:
            raise ValueError('Annotation references an unknown image')
        if not isinstance(annotation['segmentation'], list):
            raise ValueError('Expected COCO polygon annotations')
    remaining = sorted(set(fish.values()) - test_fish)
    random.Random(42).shuffle(remaining)
    val_fish = set(remaining[:round(len(remaining) * 0.2)])
    splits = {key: [] for key in ('train', 'val', 'test')}
    for name in sorted(names):
        key = 'test' if fish[name] in test_fish else 'val' if fish[name] in val_fish else 'train'
        splits[key].append(name)
    if any(not values for values in splits.values()):
        raise ValueError('All three splits must be non-empty')
    groups = {key: {fish[name] for name in values} for key, values in splits.items()}
    for left, right in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        if groups[left] & groups[right]:
            raise ValueError('Fish overlap between splits')
    if set(splits['test']) != {name for name in names if fish[name] in test_fish}:
        raise ValueError('Test split differs from common test fish')
    return source, coco, splits


def synchronize(debut_root, destination, dry_run=False):
    source, coco, splits = prepare(debut_root)
    print(f"{len(coco['images'])} images: " + ', '.join(
        f'{key}={len(values)}' for key, values in splits.items()))
    if dry_run:
        return
    destination.mkdir(parents=True, exist_ok=True)
    # Stage all copies before touching the existing image directory.
    with tempfile.TemporaryDirectory(prefix='.sync-', dir=destination) as temporary:
        stage = Path(temporary)
        (stage / 'images').mkdir()
        for image in coco['images']:
            name = image['file_name']
            shutil.copy2(source / name, stage / 'images' / name)
        payloads = {
            '_annotations.coco.json': coco,
            'splits.json': splits,
            'splits_smoke.json': {key: values[:1] for key, values in splits.items()},
        }
        for name, payload in payloads.items():
            (stage / name).write_text(json.dumps(payload, indent=2) + '\n')
        previous = destination / 'images'
        if previous.exists() or previous.is_symlink():
            previous.rename(stage / 'previous_images')
        (stage / 'images').rename(previous)
        for name in payloads:
            (stage / name).replace(destination / name)
    print(f'Synchronized {destination}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--debut-root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--dry-run', action='store_true', help='Validate inputs and report split sizes only')
    args = parser.parse_args()
    synchronize(args.debut_root.resolve(), Path(__file__).resolve().parent / 'data', args.dry_run)
