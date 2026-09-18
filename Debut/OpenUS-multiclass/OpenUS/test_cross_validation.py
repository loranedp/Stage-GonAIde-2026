"""CPU-only checks: python3 -m unittest test_cross_validation -v."""
import ast
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import run_cross_validation as runner
from prepare_cross_validation import prepare_splits, fish_id, write_json


class CrossValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        names = [f'img_echo_{fish}_{i}.png' for fish in range(7) for i in range(2)]
        for name in names:
            (self.root / name).touch()
        self.splits = {'train': names[:8], 'val': names[8:12], 'test': names[12:]}
        self.split = self.root / 'splits.json'
        self.coco = self.root / 'annotations.json'
        self.weights = self.root / 'weights.pth'
        self.weights.write_bytes(b'weights')
        write_json(self.split, self.splits)
        write_json(self.coco, {'images': [{'file_name': n} for n in names]})

    def prepare(self):
        return prepare_splits(self.split, self.coco, self.root)

    def test_coverage_and_reproducibility(self):
        folds, manifest = self.prepare()
        self.assertEqual((folds, manifest), self.prepare())
        self.assertEqual(sorted(n for f in folds for n in f['val']),
                         sorted(self.splits['train'] + self.splits['val']))
        for fold in folds:
            self.assertFalse({fish_id(n) for n in fold['train']} & {fish_id(n) for n in fold['val']})
            self.assertEqual(fold['test'], self.splits['test'])

    def test_invalid_splits(self):
        original = json.loads(self.split.read_text())
        for bad in [dict(original, train=original['train'] * 2),
                    dict(original, val=original['train']),
                    dict(original, train=['img_echo_99_0.png'])]:
            write_json(self.split, bad)
            with self.assertRaises(ValueError):
                self.prepare()
        write_json(self.split, dict(original, train=original['train'][:2], val=original['val'][:2]))
        with self.assertRaises(ValueError):
            self.prepare()
        write_json(self.split, original)
        (self.root / original['train'][0]).unlink()
        with self.assertRaises(FileNotFoundError):
            self.prepare()

    def test_checkpoint_selection(self):
        tree = ast.parse((runner.ROOT / 'eval_segmentation.py').read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'checkpoint_improved')
        namespace = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<selection>', 'exec'), namespace)
        select = namespace['checkpoint_improved']
        self.assertTrue(select('dice', 0, 0, 0, 0, False))
        self.assertFalse(select('dice', .9, .4, .5, .4, True))
        self.assertTrue(select('dice', .3, .6, .5, .4, True))
        self.assertTrue(select('either', .9, .4, .5, .4, True))

    def test_aggregation(self):
        rows = [{'validation': {'loss': i, 'miou': i, 'dice': i},
                 'test': {'loss': i, 'miou': i, 'dice': i}} for i in range(5)]
        result = runner.aggregate(rows)['test']['dice']
        self.assertEqual(result['mean'], 2)
        self.assertAlmostEqual(result['std'], 2.5 ** .5)
        with self.assertRaises(ValueError):
            runner.aggregate(rows[:4])
        bad = self.root / 'bad.json'
        write_json(bad, {'test_metrics': {'loss': 'bad', 'miou': 0, 'dice': 0}})
        with self.assertRaises(ValueError):
            runner.read_metrics(bad, 'test_metrics')

    def test_orchestration_and_resume(self):
        output = self.root / 'output'
        args = ['--split_file', str(self.split), '--coco_json', str(self.coco),
                '--images_root', str(self.root), '--pretrained_weights', str(self.weights),
                '--output_dir', str(output)]
        calls = []
        fail_test = [True]

        def fake(command, log):
            calls.append(command)
            def value(key):
                return command[command.index(key) + 1]
            attempt = Path(value('--output_dir'))
            metrics = {'loss': .3, 'miou': .4, 'dice': .5,
                       'iou_per_class': [.4] * 3, 'dice_per_class': [.5] * 3}
            if 'eval_segmentation.py' in command[1]:
                (attempt / 'checkpoint_teacher_seg_best.pth').touch()
                write_json(attempt / 'validation_results_teacher_best.json',
                           {'epoch': 1, 'validation_metrics': metrics})
            elif '--eval_split' in command:
                dest = Path(value('--results_dir'))
                masks = dest / 'predicted_masks_teacher'
                masks.mkdir(parents=True, exist_ok=True)
                for name in json.loads(Path(value('--split_file')).read_text())['val']:
                    (masks / f'pred_{Path(name).stem}.txt').touch()
                write_json(dest / 'test_results_teacher_best.json', {'test_metrics': metrics})
            else:
                if fail_test[0]:
                    fail_test[0] = False
                    raise RuntimeError('simulated failure')
                write_json(attempt / 'test_results_teacher_best.json', {'test_metrics': metrics})

        with patch.object(runner, 'run_command', fake), contextlib.redirect_stdout(io.StringIO()):
            runner.main(args + ['--dry-run'])
            self.assertFalse(output.exists())
            with self.assertRaises(RuntimeError):
                runner.main(args)
            self.assertEqual(len(calls), 3)
            runner.main(args + ['--resume'])
            self.assertEqual(len(calls), 16)
            runner.main(args + ['--resume'])
            self.assertEqual(len(calls), 16)
            with self.assertRaises(ValueError):
                runner.main(args + ['--resume', '--epochs', '2'])
        self.assertTrue((output / 'summary.csv').exists())
        self.assertEqual(len(json.loads((output / 'summary.json').read_text())['folds']), 5)


if __name__ == '__main__':
    unittest.main()
