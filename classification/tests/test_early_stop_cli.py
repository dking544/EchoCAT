import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
import train


class EarlyStopCLITests(unittest.TestCase):
    def exercise(self, task, values, monitor_name, expected_stop, expected_best):
        with tempfile.TemporaryDirectory(prefix='echocat_monitor_') as tmp:
            root = Path(tmp)
            classes = len(train.CLASSES[task])
            for i in range(classes * 2):
                (root / f'{i}.png').touch()
            train_list, val_list = root/'train.txt', root/'val.txt'
            train_list.write_text(''.join(f'{i}.png {i}\n' for i in range(classes)))
            val_list.write_text(''.join(f'{i+classes}.png {i}\n' for i in range(classes)))
            mapping = root/'cases.jsonl'
            mapping.write_text('\n'.join(json.dumps({'image': f'{i+classes}.png',
                'fetus_id': f'example_{i}'}) for i in range(classes)))
            models = []

            def build(*args, **kwargs):
                model = torch.nn.Linear(1, 1, bias=False)
                models.append(model)
                return model

            def fit(model, runner, scheduler, device, epoch, *args):
                with torch.no_grad():
                    model.weight.fill_(epoch+1)
                runner['iter'] += len(runner['train_loader'])

            def validate(model, *args):
                value = values[int(model.weight.item())-1]
                return {'accuracy': value, 'precision_macro': value,
                        'f1_macro': value, 'support_per_class': [1]*classes,
                        'fetal': {'f1_macro': value}}

            base = ['train.py', '--task', task, '--train-list', str(train_list),
                    '--val-list', str(val_list), '--device', 'cpu', '--workers', '0',
                    '--batch-size', '2', '--epochs', str(len(values)),
                    '--early-stop-monitor', monitor_name]
            if task == 'six':
                base += ['--val-case-map', str(mapping)]
            for name in ['first', 'resumed']:
                output = root/name
                argv = base+['--output', str(output)]
                if name == 'resumed':
                    argv += ['--resume-best', str(root/'first'/f'epoch_{expected_best}.pth')]
                with patch.object(sys, 'argv', argv), patch.object(train, 'build', build), \
                     patch.object(train.legacy, 'train', fit), patch.object(train, 'validate', validate), \
                     patch('echocat.fetal_validation.validate_fetal', validate), \
                     contextlib.redirect_stdout(io.StringIO()):
                    train.main()
                summary = json.loads((output/'early_stop_summary.json').read_text())
                self.assertEqual(summary['last_epoch'], expected_stop)
                self.assertEqual(summary['restored_epoch'], expected_best)
                self.assertTrue(summary['early_stopped'])
                self.assertEqual(models[-1].weight.item(), expected_best)
                checkpoint = torch.load(output/f'epoch_{expected_best}.pth', weights_only=True)
                self.assertEqual(checkpoint['epoch'], expected_best)
                self.assertEqual(checkpoint['state_dict']['weight'].item(), expected_best)
                self.assertEqual({x.name for x in output.glob('*.pth')},
                                 {f'epoch_{expected_best}.pth',f'epoch_{expected_stop}.pth'})
                index = json.loads((output/'checkpoints.json').read_text())
                self.assertEqual(index['best_epoch'],expected_best)
                self.assertEqual(index['last_epoch'],expected_stop)
                last = torch.load(output/f'epoch_{expected_stop}.pth', weights_only=True)
                self.assertEqual(last['epoch'], expected_stop)
                self.assertEqual(last['state_dict']['weight'].item(), expected_stop)
                self.assertEqual(last['checkpoint_kind'], 'last')
                self.assertEqual(last['monitor_state']['wait'], 10 if task=='binary' else 15)
                last_record = json.loads((output/'last_validation.json').read_text())
                self.assertEqual(last_record['epoch'], expected_stop)
                self.assertFalse(last_record['is_best'])
                bad_argv = base+['--output',str(root/'invalid_resume'),
                                '--resume-best',str(output/f'epoch_{expected_stop}.pth')]
                with patch.object(sys,'argv',bad_argv), patch.object(train,'build',build):
                    with self.assertRaisesRegex(ValueError,'requires the selected best checkpoint'):
                        train.main()

    def test_binary_stop_restore_and_resume(self):
        self.exercise('binary', [.1,.2,.3]+[0.]*15, 'macro_precision_3mean', 13, 3)

    def test_fetal_stop_restore_and_resume(self):
        self.exercise('six', [.4,.5]+[.1]*18, 'fetal_macro_f1', 17, 2)

    def test_invalid_cli_combinations(self):
        base = ['train.py', '--train-list', 'unused', '--val-list', 'unused', '--output', 'unused']
        cases = [
            ['--task','six','--early-stop-monitor','fetal_macro_f1'],
            ['--task','binary','--early-stop-monitor','fetal_macro_f1'],
            ['--task','view','--early-stop-monitor','macro_precision_3mean'],
            ['--task','binary','--early-stop-monitor','macro_precision_3mean','--epochs','2'],
            ['--task','binary','--early-stop-monitor','macro_precision_3mean','--selection','accuracy'],
            ['--task','six','--val-case-map','unused'],
        ]
        for extra in cases:
            with patch.object(sys, 'argv', base+extra), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    train.main()
                self.assertEqual(caught.exception.code, 2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
