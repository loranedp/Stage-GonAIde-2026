"""Prepare reproducible five-fold splits grouped by fish (standard library only)."""
import argparse
import hashlib
import json
import random
from pathlib import Path


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def fish_id(name):
    if Path(name).name != name or len(name.split('_')) < 4:
        raise ValueError(f'Invalid image name: {name}')
    return name.split('_')[2]


def prepare_splits(split_file, coco_json, images_root, seed=42):
    splits = json.loads(Path(split_file).read_text())
    coco = json.loads(Path(coco_json).read_text())
    names = [image['file_name'] for image in coco['images']]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate COCO image names')
    available = set(names)
    seen = set()
    for key in ('train', 'val', 'test'):
        values = splits[key]
        if not isinstance(values, list) or len(values) != len(set(values)):
            raise ValueError(f'Invalid or duplicate images in {key}')
        if seen.intersection(values):
            raise ValueError('Image overlap between source splits')
        seen.update(values)
        for name in values:
            fish_id(name)
            if name not in available:
                raise ValueError(f'Unknown COCO image: {name}')
            if not (Path(images_root) / name).is_file():
                raise FileNotFoundError(Path(images_root) / name)
    pool = sorted(splits['train'] + splits['val'])
    fish = sorted({fish_id(name) for name in pool})
    test_fish = {fish_id(name) for name in splits['test']}
    if not splits['test'] or set(fish) & test_fish:
        raise ValueError('Test must be nonempty and contain separate fish')
    if len(fish) < 5:
        raise ValueError('At least five train/validation fish are required')
    random.Random(seed).shuffle(fish)
    folds = []
    counts = []
    for index in range(5):
        validation_fish = set(fish[index::5])
        fold = {
            'train': [name for name in pool if fish_id(name) not in validation_fish],
            'val': [name for name in pool if fish_id(name) in validation_fish],
            'test': list(splits['test']),
        }
        if not fold['train'] or not fold['val']:
            raise ValueError('Empty training or validation fold')
        folds.append(fold)
        counts.append({key: {'images': len(values),
                             'fish': sorted({fish_id(name) for name in values})}
                       for key, values in fold.items()})
    validation = [name for fold in folds for name in fold['val']]
    if len(validation) != len(pool) or set(validation) != set(pool):
        raise ValueError('Invalid validation coverage')
    manifest = {
        'seed': seed, 'folds': counts,
        'source_sha256': {key: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                          for key, path in [('splits', split_file), ('annotations', coco_json)]},
        'folds_sha256': hashlib.sha256(json.dumps(folds, sort_keys=True).encode()).hexdigest(),
    }
    return folds, manifest


def save_splits(destination, folds, manifest):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for index, fold in enumerate(folds, 1):
        write_json(destination / f'fold_{index}.json', fold)
    write_json(destination / 'manifest.json', manifest)


if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    debut_data = root.parents[1] / 'data'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split_file', type=Path, default=root / 'data/splits.json')
    parser.add_argument('--coco_json', type=Path,
                        default=debut_data / 'COCO/labels/cavite/annotations.json')
    parser.add_argument('--images_root', type=Path,
                        default=debut_data / 'COCO/images/cavite')
    parser.add_argument('--output_dir', type=Path, default=root / 'output/custom_seg_cv5/splits')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    folds, manifest = prepare_splits(args.split_file, args.coco_json, args.images_root, args.seed)
    print(json.dumps(manifest, indent=2))
    if not args.dry_run:
        if args.output_dir.exists():
            raise SystemExit('Output exists; choose another output directory')
        save_splits(args.output_dir, folds, manifest)
