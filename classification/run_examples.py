import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def check_inputs(root, expected):
    images = expected['images']
    paths = {row['image'] for row in images}
    actual = {p.relative_to(root).as_posix() for p in (root / 'examples').rglob('*.png')}
    if actual != paths or len(paths) != len(images):
        raise ValueError('Example image inventory does not match expected_results.json')
    for row in images:
        if hashlib.sha256((root / row['image']).read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('Example image checksum mismatch: ' + row['image'])
    view_lines = (root / 'examples/view.txt').read_text(encoding='utf-8').splitlines()
    if view_lines != [f"{Path(row['image']).relative_to('examples').as_posix()} {row['view_label']}" for row in images]:
        raise ValueError('View manifest does not match expected image order and labels')
    for task in ('binary', 'six'):
        rows = [json.loads(s) for s in (root / f'examples/fetal_{task}.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
        cases = {c['fetus_id']: c for c in expected['cases']}
        wanted = [{'image': Path(r['image']).relative_to('examples').as_posix(), 'fetus_id': r['fetus_id'], 'fetus_label': cases[r['fetus_id']][task]['expected_label']} for r in images]
        if rows != wanted:
            raise ValueError('Fetal manifest does not match expected image order and labels')


def main():
    parser = argparse.ArgumentParser(description='Test the bundled examples with the fixed models.')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--output-dir', default='results/examples')
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    root = Path(__file__).resolve().parent
    expected = read_json(root / 'examples/expected_results.json')
    check_inputs(root, expected)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    commands = [('view', 'predict.py', ['--task', 'view', '--list', 'examples/view.txt', '--protocol', 'external'])]
    commands += [(t, 'fetal.py', ['--task', t, '--manifest', f'examples/fetal_{t}.jsonl']) for t in ('binary', 'six')]
    for task, script, options in commands:
        print(f'Running {task}...', flush=True)
        subprocess.run([sys.executable, str(root / script), *options, '--output', str(output / f'{task}.json'), '--device', args.device, '--batch-size', str(args.batch_size)], cwd=root, check=True)
    view = read_json(output / 'view.json')
    records = view['records']
    if len(records) != len(expected['images']):
        raise ValueError('Unexpected number of view predictions')
    for record, ref in zip(records, expected['images']):
        if Path(record['image']).resolve() != (root / ref['image']).resolve() or record['label'] != ref['view_label']:
            raise ValueError('View output order or labels do not match the input manifest')
    fetal = {task: read_json(output / f'{task}.json') for task in ('binary', 'six')}
    summary = {'purpose': expected['purpose'], 'device': args.device, 'images': len(records), 'cases': [], 'all_expected_predictions_reproduced': True}
    for task in fetal:
        ids = [r['fetus_id'] for r in fetal[task]['records']]
        if len(ids) != len(expected['cases']) or set(ids) != {c['fetus_id'] for c in expected['cases']}:
            raise ValueError('Unexpected fetal output inventory')
    for case in expected['cases']:
        cid = case['fetus_id']
        indices = [i for i, row in enumerate(expected['images']) if row['fetus_id'] == cid]
        if len(indices) != case['image_count']:
            raise ValueError('Unexpected image count for ' + cid)
        correct = sum(records[i]['prediction'] == expected['images'][i]['view_label'] for i in indices)
        matched = sum(records[i]['prediction'] == expected['images'][i]['reference_view_prediction'] for i in indices)
        result = {'fetus_id': cid, 'images': len(indices), 'view_correct': correct, 'view_reference_predictions_matched': matched, 'predicted_4ch': sum(records[i]['scores'][0] >= 0.5 for i in indices)}
        summary['all_expected_predictions_reproduced'] &= matched == len(indices)
        for task in ('binary', 'six'):
            row = next(r for r in fetal[task]['records'] if r['fetus_id'] == cid)
            selected = row['selected_indices']
            if not selected or len(set(selected)) != len(selected) or not set(selected).issubset(indices):
                raise ValueError('Invalid selected images for ' + cid)
            matches = row['prediction'] == case[task]['expected_label']
            result[task] = {'expected_class': case[task]['expected_class'], 'predicted_class': fetal[task]['class_names'][row['prediction']], 'matches_expected': matches, 'scores': row['scores'], 'retained_images': len(selected), 'selected_images': [expected['images'][i]['image'] for i in selected], 'used_fallback': row['used_fallback']}
            summary['all_expected_predictions_reproduced'] &= matches
        summary['cases'].append(result)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    for case in summary['cases']:
        print(f"{case['fetus_id']}: {case['images']} images; view correct {case['view_correct']}/{case['images']}; binary {case['binary']['predicted_class']}; six {case['six']['predicted_class']}")
    if not summary['all_expected_predictions_reproduced']:
        raise SystemExit('Outputs differ from the reference. Inspect summary.json.')
    print('All reference view predictions and four final fetal diagnoses reproduced.')


if __name__ == '__main__':
    main()
