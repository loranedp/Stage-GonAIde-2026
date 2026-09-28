"""Train and test five independent OpenUS folds with one command."""
import argparse
import csv
import json
import math
import shlex
import statistics
import subprocess
import sys
import pickle
from pathlib import Path

from prepare_cross_validation import prepare_splits, save_splits, write_json

ROOT = Path(__file__).resolve().parent
DEBUT_ROOT = ROOT.parents[1]
DEBUT_DATA_ROOT = DEBUT_ROOT / 'data'


def discover_classes(coco_json):
    coco = json.loads(Path(coco_json).read_text())
    classes = [category['name'] for category in sorted(coco.get('categories', []),
                                                       key=lambda value: int(value['id']))
               if int(category['id']) > 0 and category['name'].strip().lower() != 'ignore']
    if not classes:
        raise ValueError(f'No foreground classes in {coco_json}')
    return classes


def read_metrics(path, key, class_count=None):
    value = json.loads(Path(path).read_text())[key]
    for name in ('loss', 'miou', 'dice'):
        number = value[name]
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number):
            raise ValueError(f'Invalid {name} in {path}')
    for name in ('precision', 'recall'):
        if name not in value:
            continue
        number = value[name]
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number):
            raise ValueError(f'Invalid {name} in {path}')
    for name in ('iou_per_class', 'dice_per_class', 'precision_per_class', 'recall_per_class'):
        if name in value:
            if ((class_count is not None and len(value[name]) != class_count)
                    or any(isinstance(v, bool) or not isinstance(v, (int, float))
                           or not math.isfinite(v) for v in value[name])):
                raise ValueError(f'Invalid {name} in {path}')
    if 'diff_surface_per_class' in value:
        surface = value['diff_surface_per_class']
        if ((class_count is not None and len(surface) != class_count)
                or any(item is not None and (isinstance(item, bool)
                       or not isinstance(item, (int, float)) or not math.isfinite(item))
                       for item in surface)):
            raise ValueError(f'Invalid diff_surface_per_class in {path}')
    if value.get('diff_surface') is not None:
        number = value['diff_surface']
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number):
            raise ValueError(f'Invalid diff_surface in {path}')
    return value


def aggregate(rows):
    if len(rows) != 5:
        raise ValueError('Five complete folds are required')
    summary = {}
    for section in ('validation', 'test'):
        result = {}
        for name in ('loss', 'miou', 'dice', 'precision', 'recall', 'diff_surface',
                     'iou_per_class', 'dice_per_class',
                     'precision_per_class', 'recall_per_class', 'diff_surface_per_class'):
            present = [name in row[section] for row in rows]
            if not any(present):
                continue
            if not all(present):
                raise ValueError(f'Inconsistent {section} metric: {name}')
            values = [row[section][name] for row in rows]
            if name.endswith('_per_class'):
                result[name] = []
                for class_values in zip(*values):
                    finite = [value for value in class_values
                              if value is not None and math.isfinite(value)]
                    result[name].append({
                        'mean': statistics.mean(finite) if finite else None,
                        'std': statistics.stdev(finite) if len(finite) > 1 else None,
                    })
            else:
                finite = [value for value in values
                          if value is not None and math.isfinite(value)]
                result[name] = {
                    'mean': statistics.mean(finite) if finite else None,
                    'std': statistics.stdev(finite) if len(finite) > 1 else None,
                }
        summary[section] = result
    return summary


def commands(args, fold, attempt):
    common = ['--arch', 'vmamba_small', '--dataset_name', 'CUSTOM', '--multilabel', 'True',
              '--coco_json', str(args.coco_json), '--metadata_file', str(args.metadata_file),
              '--images_root', str(args.images_root), '--split_file', str(fold),
              '--pretrained_weights', str(args.pretrained_weights), '--checkpoint_key', args.checkpoint_key,
              '--pretrained_vmamba', str(args.pretrained_vmamba), '--img_size', str(args.img_size),
              '--batch_size_per_gpu', str(args.batch_size_per_gpu), '--num_workers', str(args.num_workers),
              '--seed', str(args.seed), '--output_dir', str(attempt)]
    train = [sys.executable, str(ROOT / 'eval_segmentation.py'), *common,
             '--epochs', str(args.epochs), '--lr', str(args.lr), '--val_freq', str(args.val_freq),
             '--selection_metric', 'dice', '--log_name', 'cv']
    test = [sys.executable, str(ROOT / 'test_segmentation.py'), *common, '--cpk_name', 'best',
            '--save_predictions', str(args.save_predictions), '--save_masks', str(args.save_masks),
            '--save_overlays', str(args.save_overlays)]
    return train, test


def oof_command(test, attempt):
    return test + ["--eval_split", "val", "--results_dir", str(attempt / "eval"),
                   "--save_masks", "True", "--save_predictions", "False"]


def run_command(command, log_path):
    print(shlex.join(command), flush=True)
    with log_path.open('w') as log:
        with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
            try:
                for line in process.stdout:
                    print(line, end='', flush=True)
                    log.write(line)
                    log.flush()
                code = process.wait()
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise
    if code:
        raise RuntimeError(f'Command failed ({code}); see {log_path}')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name, default in [('split_file', ROOT / 'data/splits.json'),
                          ('coco_json', DEBUT_DATA_ROOT / 'COCO/labels/cavite/annotations.json'),
                          ('images_root', DEBUT_DATA_ROOT / 'COCO/images/cavite'),
                          ('metadata_file', DEBUT_ROOT / 'utils/correspondance_echo_final.xlsx'),
                          ('pretrained_weights', ROOT / 'checkpoint/openus_cpt0150.pth'),
                          ('output_dir', 'output/custom_seg_cv5')]:
        p.add_argument('--' + name, type=Path, default=ROOT / default if isinstance(default, str) else default)
    for name, default in [('epochs', 100), ('batch_size_per_gpu', 4), ('num_workers', 4),
                          ('img_size', 512), ('val_freq', 1), ('seed', 42)]:
        p.add_argument('--' + name, type=int, default=default)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--checkpoint_key', default='teacher')
    for name in ('pretrained_vmamba', 'save_predictions', 'save_masks', 'save_overlays'):
        p.add_argument('--' + name, action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--recompute-metrics', action='store_true',
                   help='Reuse existing checkpoints and rerun validation/test evaluation')
    p.add_argument('--dry-run', action='store_true')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, (ROOT / value).resolve())
    if min(args.epochs, args.batch_size_per_gpu, args.img_size, args.val_freq) <= 0 or args.num_workers < 0:
        raise ValueError('Epochs, batch, image size and validation frequency must be positive; workers nonnegative')
    if not math.isfinite(args.lr) or args.lr <= 0 or not args.checkpoint_key.strip() or ',' in args.checkpoint_key or Path(args.checkpoint_key).name != args.checkpoint_key:
        raise ValueError('Use a positive finite learning rate and one checkpoint key')
    folds, metadata = prepare_splits(args.split_file, args.coco_json, args.images_root, args.seed)
    class_names = discover_classes(args.coco_json)
    pool = folds[0]['train'] + folds[0]['val']
    if len({Path(name).stem for name in pool}) != len(pool):
        raise ValueError('Image stems must be unique for out-of-fold TXT export')
    if not args.pretrained_weights.is_file():
        raise FileNotFoundError(args.pretrained_weights)
    if not args.metadata_file.is_file():
        raise FileNotFoundError(args.metadata_file)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
              if key not in ('resume', 'dry_run', 'recompute_metrics')}
    metadata['configuration'] = config
    # Hash large checkpoints in chunks to avoid allocating them in memory.
    import hashlib
    digest = hashlib.sha256()
    with args.pretrained_weights.open('rb') as weights:
        for chunk in iter(lambda: weights.read(1024 * 1024), b''):
            digest.update(chunk)
    metadata['pretrained_sha256'] = digest.hexdigest()
    out = args.output_dir
    if args.dry_run:
        for index in range(1, 6):
            print(f'Fold {index}: {metadata["folds"][index - 1]}')
            attempt = out / f'fold_{index}/attempt_1'
            train, test = commands(args, out / 'splits' / f'fold_{index}.json', attempt)
            for command in (train, oof_command(test, attempt), test):
                print(shlex.join(command))
        return
    if out.exists():
        if not args.resume and not args.recompute_metrics:
            raise ValueError('Output exists; use --resume or another --output_dir')
        stored_manifest = json.loads((out / 'splits/manifest.json').read_text())
        if args.recompute_metrics:
            if stored_manifest.get('folds_sha256') != metadata.get('folds_sha256'):
                raise ValueError('Generated folds differ from the stored fold manifest')
        elif stored_manifest != metadata:
            raise ValueError('Resume configuration or input data differs from the manifest')
        for index, fold in enumerate(folds, 1):
            if json.loads((out / 'splits' / f'fold_{index}.json').read_text()) != fold:
                raise ValueError('Stored fold differs from the manifest')
    else:
        save_splits(out / 'splits', folds, metadata)
    rows = []
    for index in range(1, 6):
        print(f'\nFold {index}/5', flush=True)
        directory = out / f'fold_{index}'
        directory.mkdir(exist_ok=True)
        state_path = directory / 'state.json'
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        if state.get('stage') in ('trained', 'evaluated', 'complete'):
            attempt = directory / state['attempt']
        else:
            number = 1
            while (directory / f'attempt_{number}').exists():
                number += 1
            attempt = directory / f'attempt_{number}'
            attempt.mkdir()
            state = {'attempt': attempt.name, 'stage': 'training'}
            write_json(state_path, state)
        checkpoint = attempt / f'checkpoint_{args.checkpoint_key}_seg_best.pth'
        validation_path = attempt / f'validation_results_{args.checkpoint_key}_best.json'
        test_path = attempt / f'test_results_{args.checkpoint_key}_best.json'
        train, test = commands(args, out / 'splits' / f'fold_{index}.json', attempt)
        if state['stage'] == 'training':
            run_command(train, attempt / 'train_console.log')
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            read_metrics(validation_path, 'validation_metrics', len(class_names))
            state['stage'] = 'trained'
            write_json(state_path, state)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        validation = read_metrics(validation_path, 'validation_metrics', len(class_names))
        if state['stage'] == 'trained' or args.recompute_metrics:
            evaluation = attempt / 'eval'
            run_command(oof_command(test, attempt), attempt / 'eval_console.log')
            read_metrics(evaluation / f'test_results_{args.checkpoint_key}_best.json', 'test_metrics', len(class_names))
            mask_dir = evaluation / f'predicted_masks_{args.checkpoint_key}'
            expected = {f'pred_{Path(name).stem}.txt' for name in folds[index - 1]['val']}
            actual = {path.name for path in mask_dir.glob('*.txt')}
            if actual != expected:
                raise ValueError(f'Missing or unexpected out-of-fold TXT files in {mask_dir}')
            state['stage'] = 'evaluated'
            write_json(state_path, state)
        evaluation = attempt / 'eval'
        validation = read_metrics(
            evaluation / f'test_results_{args.checkpoint_key}_best.json',
            'test_metrics', len(class_names),
        )
        expected = {f'pred_{Path(name).stem}.txt' for name in folds[index - 1]['val']}
        actual = {path.name for path in (evaluation / f'predicted_masks_{args.checkpoint_key}').glob('*.txt')}
        if actual != expected:
            raise ValueError(f'Missing or unexpected out-of-fold TXT files in {evaluation}')
        if state['stage'] != 'complete' or args.recompute_metrics:
            if test_path.exists() and not args.recompute_metrics:
                test_path.rename(attempt / 'previous_test_results.json')
            run_command(test, attempt / 'test_console.log')
            read_metrics(test_path, 'test_metrics', len(class_names))
            state['stage'] = 'complete'
            write_json(state_path, state)
        rows.append({'fold': index, 'counts': metadata['folds'][index - 1],
                     'selected_epoch': json.loads(validation_path.read_text())['epoch'],
                     'checkpoint': str(checkpoint), 'out_of_fold_txt_dir': str(evaluation / f'predicted_masks_{args.checkpoint_key}'),
                     'validation': validation,
                     'test': read_metrics(test_path, 'test_metrics', len(class_names))})
    summary = aggregate(rows)
    write_json(out / 'summary.json', {'configuration': config, 'folds': rows, 'summary': summary,
                                     'class_order': class_names,
                                     'aggregation': 'Unweighted fold mean; sample standard deviation (ddof=1).',
                                     'metric_note': 'Native-resolution per-image metrics; fold summaries are unweighted.'})
    fields = ['fold', 'selected_epoch']
    flat_rows = []
    for row in rows:
        flat = {key: row[key] for key in fields}
        for section in ('validation', 'test'):
            for name, value in row[section].items():
                if name in ('loss', 'miou', 'dice', 'precision', 'recall', 'diff_surface'):
                    flat[f'{section}_{name}'] = value
                elif name.endswith('_per_class'):
                    for label, number in zip(class_names, value):
                        flat[f'{section}_{name}_{label}'] = number
        flat_rows.append(flat)
    with (out / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    for section, metrics in summary.items():
        print(f'\n{section}:')
        for name in ('loss', 'miou', 'dice'):
            print(f'  {name}: {metrics[name]["mean"]:.4f} ± {metrics[name]["std"]:.4f}')
    export_oof_results(out, rows, class_names, args.checkpoint_key)
    print(f'Results: {out / "summary.json"}')


def export_oof_results(out, rows, class_names, checkpoint_key):
    """Create analysis-friendly CSV and an UNet-compatible fold dictionary."""
    csv_rows = []
    pickle_result = {}
    metric_keys = {
        'iou': 'iou_by_class', 'dice': 'dice_by_class',
        'precision': 'precision_by_class', 'recall': 'recall_by_class',
        'diff_surface': 'diff_surface',
    }
    for fold_index, row in enumerate(rows, 1):
        result_path = (Path(row['out_of_fold_txt_dir']).parent /
                       f'test_results_{checkpoint_key}_best.json')
        payload = json.loads(result_path.read_text())
        images = payload.get('per_image', [])
        fold = {'name': [image['file_name'] for image in images]}
        for output_key in metric_keys.values():
            fold[output_key] = {class_name: [] for class_name in class_names}
        for image in images:
            flat = {'file_name': image['file_name'], 'fold': fold_index}
            by_class = image.get('metrics_by_class', {})
            for class_name in class_names:
                metrics = by_class[class_name]
                label = class_name.lower().replace(' ', '_')
                for metric, output_key in metric_keys.items():
                    value = metrics[metric]
                    fold[output_key][class_name].append(
                        float('nan') if value is None else value)
                    flat[f'{metric}_{label}' + ('_cm2' if metric == 'diff_surface' else '')] = value
            csv_rows.append(flat)
        pickle_result[f'fold_{fold_index}'] = fold

    csv_path = out / 'oof_validation_metrics.csv'
    with csv_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    suffix = f'{len(class_names)}classes'
    with (out / f'results_OpenUS_{suffix}.pkl').open('wb') as stream:
        pickle.dump(pickle_result, stream)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, KeyError, OSError, RuntimeError) as error:
        raise SystemExit(str(error))
