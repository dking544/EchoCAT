from pathlib import Path
import copy
import json
import torch
from models.build import BuildNet
from echocat import transforms

ROOT = Path(__file__).resolve().parents[1]
CLASSES = {
    'view': ['4CH', 'Non-4CH'],
    'binary': ['Normal', 'Abnormal'],
    'six': ['Normal', 'Single Ventricle', 'Septal Defect', 'Ebstein Anomaly',
            'Cardiac Tumor', 'Ventricular Disproportion'],
}


def recipe(task):
    if task not in CLASSES:
        raise ValueError(f'Unknown task: {task}')
    cfg = json.loads((ROOT / 'recipes' / f'{task}.json').read_text())
    cfg['data']['test']['metric_options']['topk'] = tuple(cfg['data']['test']['metric_options']['topk'])
    cfg['optimizer']['betas'] = tuple(cfg['optimizer']['betas'])
    for pipeline in (cfg['train_pipeline'], cfg['val_pipeline']):
        for step in pipeline:


            if isinstance(step.get('size'), list):
                step['size'] = tuple(step['size'])
            if hasattr(transforms, step['type']):
                step['type'] = getattr(transforms, step['type'])
    if task != 'view':
        cfg['model']['cam_generator']['checkpoint'] = str(ROOT / 'weights' / f'{task}_teacher.pth')
    return cfg


def build(task, checkpoint=None, device='cpu', initialize=False, teacher=None):
    cfg = recipe(task)
    if teacher:
        if task == 'view':
            raise ValueError('View classifier does not use a CAM teacher')
        cfg['model']['cam_generator']['checkpoint'] = str(Path(teacher).resolve())
    model = BuildNet(copy.deepcopy(cfg['model']))
    if initialize:
        model.init_weights()
    if checkpoint is not None:
        payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
        state = payload.get('state_dict', payload)

        model.load_state_dict(state, strict=True)
    return model.to(device)


def fixed_checkpoint(task):
    return ROOT / 'weights' / f'{task}_fixed.pth'


def load_lines(path, num_classes):
    path = Path(path).resolve()
    result = []
    for line_number, raw in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        if not raw.strip():
            continue
        try:
            image, label = raw.strip().rsplit(' ', 1)
            label = int(label)
        except ValueError as exc:
            raise ValueError(f'Invalid manifest line {line_number}') from exc
        if not 0 <= label < num_classes:
            raise ValueError(f'Invalid label at line {line_number}: {label}')
        image = Path(image).expanduser()
        if not image.is_absolute():
            image = path.parent / image
        image = image.resolve()
        if not image.is_file():
            raise FileNotFoundError(f'Missing image at manifest line {line_number}: {image}')
        result.append(f'{image} {label}\n')
    if not result:
        raise ValueError('Manifest contains no images')
    paths = [x.rsplit(' ', 1)[0] for x in result]
    if len(paths) != len(set(paths)):
        raise ValueError('Duplicate image paths in manifest; review the input list')
    return result
