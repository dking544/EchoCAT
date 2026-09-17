import json
import os
from pathlib import Path
import torch
import torch.nn.functional as F
from echocat.metrics import classification_metrics


def path_key(path):
    return os.path.normcase(str(Path(path).resolve()))


def load_case_map(source, validation_lines):
    source = Path(source).resolve()
    labels = {path_key(line.strip().rsplit(' ', 1)[0]): int(line.strip().rsplit(' ', 1)[1])
              for line in validation_lines}
    mapping, fetal_labels = {}, {}
    for number, text in enumerate(source.read_text(encoding='utf-8-sig').splitlines(), 1):
        if not text.strip():
            continue
        row = json.loads(text)
        if not isinstance(row, dict) or not isinstance(row.get('image'), str):
            raise ValueError(f'Case-map line {number} requires an image path')
        fetus = row.get('fetus_id')
        if not isinstance(fetus, str) or not fetus or fetus != fetus.strip():
            raise ValueError(f'Case-map line {number} requires a nonempty fetus_id')
        path = Path(row['image']).expanduser()
        key = path_key(path if path.is_absolute() else source.parent / path)
        if key not in labels:
            raise ValueError(f'Case-map line {number} is not in the validation list')
        if key in mapping:
            raise ValueError(f'Duplicate image at case-map line {number}')
        label = labels[key]
        if 'fetus_label' in row and (type(row['fetus_label']) is not int or row['fetus_label'] != label):
            raise ValueError(f'Fetal label disagrees with validation label at line {number}')
        if fetus in fetal_labels and fetal_labels[fetus] != label:
            raise ValueError('Images of one fetus have conflicting diagnostic labels')
        mapping[key], fetal_labels[fetus] = fetus, label
    if set(mapping) != set(labels):
        raise ValueError('Case map must cover every validation image exactly once')
    return mapping


def pooled_metrics(outputs, targets, paths, case_map):
    if outputs.ndim != 2 or outputs.shape[1] != 6 or len(outputs) != len(paths) or len(targets) != len(paths):
        raise ValueError('Expected six-class outputs and one path per image')
    keys = [path_key(path) for path in paths]
    if len(set(keys)) != len(keys) or set(keys) != set(case_map):
        raise ValueError('Validation loader and case map do not match exactly')
    probabilities = torch.softmax(outputs, dim=1)
    grouped, labels = {}, {}
    for i, (key, label) in enumerate(zip(keys, targets.tolist())):
        fetus = case_map[key]
        if fetus in labels and labels[fetus] != label:
            raise ValueError('Conflicting fetal validation labels')
        grouped.setdefault(fetus, []).append(probabilities[i])
        labels[fetus] = label
    ids = sorted(grouped)
    scores = torch.stack([torch.stack(grouped[fetus]).mean(dim=0) for fetus in ids])
    result = classification_metrics(scores.numpy(), [labels[fetus] for fetus in ids])
    if any(n == 0 for n in result['support_per_class']):
        raise ValueError('Fetal validation set must contain all six classes')
    result['evaluation_level'] = 'fetus_mean_probability_legacy_extra_softmax'
    return result


@torch.no_grad()
def validate_fetal(model, loader, device, case_map):
    model.eval()
    outputs, targets, paths, loss_sum = [], [], [], 0.
    for images, labels, batch_paths in loader:
        labels_device = labels.to(device)
        prediction = model(images.to(device), return_loss=False)
        loss_sum += float(F.cross_entropy(prediction, labels_device))
        outputs.append(prediction.cpu())
        targets.append(labels.cpu())
        paths.extend(batch_paths)
    if not outputs:
        raise ValueError('Validation loader has no batches')
    outputs, targets = torch.cat(outputs), torch.cat(targets)
    result = classification_metrics(outputs.numpy(), targets.numpy())
    result['avg_loss'] = loss_sum / len(loader)
    result['fetal'] = pooled_metrics(outputs, targets, paths, case_map)
    result['validation_protocol'] = 'historical_fetal_equal_mean'
    return result
