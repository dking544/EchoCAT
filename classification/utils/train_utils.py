import os
import torch
import torch.distributed as dist
import sys
import types
import importlib
import random
from tqdm import tqdm
import numpy as np
from numpy import mean
import torch.nn.functional as F
try:
    from terminaltables import AsciiTable
except ImportError:
    class AsciiTable:


        def __init__(self, table_data, title=None):
            rows = [[str(cell) for cell in row] for row in table_data]
            widths = [
                max(len(row[index]) for row in rows)
                for index in range(len(rows[0]))
            ]
            lines = []
            if title:
                lines.append(str(title))
            for row_index, row in enumerate(rows):
                lines.append(" | ".join(
                    cell.ljust(widths[index])
                    for index, cell in enumerate(row)
                ))
                if row_index == 0:
                    lines.append("-+-".join("-" * width for width in widths))
            self.table = "\n".join(lines)
from torch.optim import Optimizer
from torch.nn.parallel import DataParallel
from core.evaluations import evaluate
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.common import get_dist_info


try:
    from sklearn.metrics import (
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
        accuracy_score
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("[Warning] sklearn not available. Enhanced metrics will be disabled.")


def init_random_seed(seed=None, device='cuda'):


    if seed is not None:
        return seed


    rank, world_size = get_dist_info()
    seed = np.random.randint(2**31)
    if world_size == 1:
        return seed

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)
    dist.broadcast(random_num, src=0)
    return random_num.item()


def set_random_seed(seed, deterministic=False):


    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def file2dict(filename):
    (path, file) = os.path.split(filename)

    abspath = os.path.abspath(os.path.expanduser(path))
    sys.path.insert(0, abspath)
    mod = importlib.import_module(file.split('.')[0])
    sys.path.pop(0)
    cfg_dict = {
                name: value
                for name, value in mod.__dict__.items()
                if not name.startswith('__')
                and not isinstance(value, types.ModuleType)
                and not isinstance(value, types.FunctionType)
                    }
    return cfg_dict.get('model_cfg'), cfg_dict.get('train_pipeline'), cfg_dict.get('val_pipeline'), cfg_dict.get('data_cfg'), cfg_dict.get('lr_config'), cfg_dict.get('optimizer_cfg')


def print_info(cfg):
    backbone = cfg.get('backbone').get('type') if cfg.get('backbone') is not None else 'None'

    if isinstance(cfg.get('neck'), list):
        temp = []
        lists = cfg.get('neck')
        for i in lists:
            temp.append(i.get('type'))
        neck = ' '.join(temp)
    else:
        neck = cfg.get('neck').get('type') if cfg.get('neck') is not None else 'None'

    head = cfg.get('head').get('type') if cfg.get('head') is not None else 'None'
    loss = cfg.get('head').get('loss').get('type') if cfg.get('head').get('loss') is not None else 'None'

    TITLE = 'Model info'
    TABLE_DATA = (
    ('Backbone', 'Neck', 'Head', 'Loss'),
    (backbone, neck, head, loss))

    table_instance = AsciiTable(TABLE_DATA, TITLE)
    print()
    print(table_instance.table)
    print()


def get_info(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    names = []
    indexs = []
    for data in class_names:
        name, index = data.split(' ')
        names.append(name)
        indexs.append(int(index))

    return names, indexs


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def resume_model(model, runner, checkpoint, meta, resume_optimizer=True, map_location='default'):
    if map_location == 'default':
        if torch.cuda.is_available():
            device_id = torch.cuda.current_device()
            checkpoint = load_checkpoint(
                model,
                checkpoint,
                map_location=lambda storage, loc: storage.cuda(device_id))
        else:
            checkpoint = load_checkpoint(model, checkpoint)
    else:
        checkpoint = load_checkpoint(
            model, checkpoint, map_location=map_location)

    runner['epoch'] = checkpoint['meta']['epoch']
    runner['iter'] = checkpoint['meta']['iter']
    runner['best_train_weight'] = checkpoint['meta']['best_train_weight']
    runner['last_weight'] = checkpoint['meta']['last_weight']
    runner['best_val_weight'] = checkpoint['meta']['best_val_weight']
    runner['best_train_loss'] = checkpoint['meta']['best_train_loss']
    runner['best_val_acc'] = checkpoint['meta']['best_val_acc']
    if meta is None:
        meta = {}


    meta = checkpoint['meta']

    if 'optimizer' in checkpoint and resume_optimizer:
        if isinstance(runner['optimizer'], Optimizer):
            runner['optimizer'].load_state_dict(checkpoint['optimizer'])
        elif isinstance(runner['optimizer'], dict):
            for k in runner['optimizer'].keys():
                runner.optimizer[k].load_state_dict(
                    checkpoint['optimizer'][k])
        else:
            raise TypeError(
                'Optimizer should be dict or torch.optim.Optimizer '
                f'but got {type(runner.optimizer)}')

    print('resumed epoch %d, iter %d' % (runner['epoch'], runner['iter']))
    return model, runner, meta


def calculate_detailed_metrics(preds_tensor, targets_tensor):


    if not SKLEARN_AVAILABLE:
        return {}


    if preds_tensor.dim() > 1:
        preds = torch.argmax(preds_tensor, dim=1).cpu().numpy()
    else:
        preds = preds_tensor.cpu().numpy()

    targets = targets_tensor.cpu().numpy()


    accuracy = accuracy_score(targets, preds)


    precision_binary = precision_score(targets, preds, average='binary', pos_label=1, zero_division=0)
    recall_binary = recall_score(targets, preds, average='binary', pos_label=1, zero_division=0)
    f1_binary = f1_score(targets, preds, average='binary', pos_label=1, zero_division=0)


    precision_macro = precision_score(targets, preds, average='macro', zero_division=0)
    recall_macro = recall_score(targets, preds, average='macro', zero_division=0)
    f1_macro = f1_score(targets, preds, average='macro', zero_division=0)


    precision_weighted = precision_score(targets, preds, average='weighted', zero_division=0)
    recall_weighted = recall_score(targets, preds, average='weighted', zero_division=0)
    f1_weighted = f1_score(targets, preds, average='weighted', zero_division=0)


    precision_per_class = precision_score(targets, preds, average=None, labels=[0, 1], zero_division=0)
    recall_per_class = recall_score(targets, preds, average=None, labels=[0, 1], zero_division=0)
    f1_per_class = f1_score(targets, preds, average=None, labels=[0, 1], zero_division=0)


    cm = confusion_matrix(targets, preds, labels=[0, 1])


    unique, counts = np.unique(targets, return_counts=True)
    class_distribution = dict(zip(unique.tolist(), counts.tolist()))

    metrics = {
        'accuracy': accuracy,

        'precision': precision_binary,
        'recall': recall_binary,
        'f1_score': f1_binary,

        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,

        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,

        'precision_per_class': precision_per_class.tolist(),
        'recall_per_class': recall_per_class.tolist(),
        'f1_per_class': f1_per_class.tolist(),
        'confusion_matrix': cm.tolist(),
        'class_distribution': class_distribution,
        'total_samples': len(targets)
    }

    return metrics


def print_detailed_metrics(metrics, title="Detailed Metrics", show_per_class=True):


    if not metrics:
        return

    print(f"\n{'='*80}")
    print(f"{title:^80}")
    print(f"{'='*80}")


    TITLE = 'Overall Metrics'
    TABLE_DATA = [
        ('Metric', 'Macro Avg', 'Weighted Avg'),
        ('Accuracy', f"{metrics['accuracy']*100:.2f}%", '-'),
        ('Precision', f"{metrics['precision_macro']*100:.2f}%", f"{metrics['precision_weighted']*100:.2f}%"),
        ('Recall', f"{metrics['recall_macro']*100:.2f}%", f"{metrics['recall_weighted']*100:.2f}%"),
        ('F1-Score', f"{metrics['f1_macro']*100:.2f}%", f"{metrics['f1_weighted']*100:.2f}%"),
    ]
    table_instance = AsciiTable(TABLE_DATA, TITLE)
    print(table_instance.table)


    if show_per_class and 'precision_per_class' in metrics:
        print(f"\n{'Per-Class Metrics':^80}")
        print('-'*80)

        num_classes = len(metrics['precision_per_class'])
        per_class_data = [('Class', 'Samples', 'Precision', 'Recall', 'F1-Score')]

        for i in range(num_classes):
            samples = metrics['class_distribution'].get(i, 0)
            per_class_data.append((
                f"Class {i}",
                str(samples),
                f"{metrics['precision_per_class'][i]*100:.2f}%",
                f"{metrics['recall_per_class'][i]*100:.2f}%",
                f"{metrics['f1_per_class'][i]*100:.2f}%"
            ))

        table_instance = AsciiTable(per_class_data, "Per-Class Metrics")
        print(table_instance.table)

    print(f"{'='*80}\n")


def _apply_batch_mix(images, targets, cfg):


    num_classes = int(cfg.get('num_classes', 2))
    if num_classes <= 1:
        raise ValueError('batch_mix.num_classes must be greater than one')
    if targets.ndim != 1:
        raise ValueError('batch_mix expects one-dimensional hard labels')

    soft_targets = F.one_hot(
        targets.long(), num_classes=num_classes).to(images.dtype)
    probability = float(cfg.get('probability', 1.0))
    if not 0.0 <= probability <= 1.0:
        raise ValueError('batch_mix.probability must be in [0, 1]')
    if images.shape[0] < 2 or np.random.random() >= probability:
        return images, soft_targets, 'none', 1.0

    mixup_alpha = float(cfg.get('mixup_alpha', 0.0))
    cutmix_alpha = float(cfg.get('cutmix_alpha', 0.0))
    switch_probability = float(cfg.get('switch_probability', 0.5))
    if mixup_alpha < 0.0 or cutmix_alpha < 0.0:
        raise ValueError('batch_mix alpha values must be non-negative')
    if mixup_alpha == 0.0 and cutmix_alpha == 0.0:
        raise ValueError('batch_mix requires a positive alpha')
    if not 0.0 <= switch_probability <= 1.0:
        raise ValueError('batch_mix.switch_probability must be in [0, 1]')

    if cutmix_alpha > 0.0 and mixup_alpha > 0.0:
        use_cutmix = np.random.random() < switch_probability
    else:
        use_cutmix = cutmix_alpha > 0.0
    alpha = cutmix_alpha if use_cutmix else mixup_alpha
    lam = float(np.random.beta(alpha, alpha))
    permutation = torch.randperm(images.shape[0], device=images.device)

    if use_cutmix:
        height, width = images.shape[-2:]
        cut_ratio = float(np.sqrt(1.0 - lam))
        cut_width = int(width * cut_ratio)
        cut_height = int(height * cut_ratio)
        center_x = int(np.random.randint(0, width))
        center_y = int(np.random.randint(0, height))
        x1 = max(center_x - cut_width // 2, 0)
        x2 = min(center_x + cut_width // 2, width)
        y1 = max(center_y - cut_height // 2, 0)
        y2 = min(center_y + cut_height // 2, height)
        mixed_images = images.clone()
        mixed_images[:, :, y1:y2, x1:x2] = images[
            permutation, :, y1:y2, x1:x2]
        lam = 1.0 - ((x2 - x1) * (y2 - y1) / float(width * height))
        method = 'cutmix'
    else:
        mixed_images = (
            images * lam + images[permutation] * (1.0 - lam))
        method = 'mixup'

    mixed_targets = (
        soft_targets * lam + soft_targets[permutation] * (1.0 - lam))
    return mixed_images, mixed_targets, method, lam


def train(model, runner, lr_update_func, device, epoch, epoches, test_cfg, meta):
    train_loss = 0
    pred_list, target_list = [], []
    runner['epoch'] = epoch + 1
    meta['epoch'] = runner['epoch']

    model.train()
    with tqdm(total=len(runner.get('train_loader')), desc=f'Train: Epoch {epoch + 1}/{epoches}', postfix=dict, mininterval=0.3) as pbar:
        for iter, batch in enumerate(runner.get('train_loader')):
            images, targets, _ = batch
            with torch.no_grad():
                images = images.to(device)
                targets = targets.to(device)
                target_list.append(targets)

                model_targets = targets
                mix_method = 'none'
                mix_lambda = 1.0
                batch_mix_cfg = runner.get('batch_mix')
                if batch_mix_cfg and batch_mix_cfg.get('enabled', False):
                    images, model_targets, mix_method, mix_lambda = (
                        _apply_batch_mix(images, targets, batch_mix_cfg))
                    state = runner.setdefault(
                        'batch_mix_state',
                        {'none': 0, 'mixup': 0, 'cutmix': 0})
                    state[mix_method] += 1

            runner.get('optimizer').zero_grad()
            lr_update_func.before_train_iter(runner)
            preds, losses = model(
                images, targets=model_targets, return_loss=True,
                train_statu=True)
            losses.get('loss').backward()
            runner.get('optimizer').step()


            pred_list.append(preds.detach())
            train_loss += losses.get('loss').item()
            postfix = {
                'Loss': train_loss / (iter + 1),
                'Cls': losses.get(
                    'classification_loss', losses.get('loss')).item(),
                'Rec': losses.get(
                    'paper_reconstruction_loss',
                    losses.get('loss').new_zeros(())).item(),
                'Lr': get_lr(runner.get('optimizer'))
            }
            if batch_mix_cfg and batch_mix_cfg.get('enabled', False):
                postfix['Mix'] = mix_method
                postfix['Lam'] = f'{mix_lambda:.3f}'
            pbar.set_postfix(**postfix)
            runner['iter'] += 1
            meta['iter'] = runner['iter']
            pbar.update(1)


    eval_results = evaluate(torch.cat(pred_list), torch.cat(target_list), test_cfg.get('metrics'), test_cfg.get('metric_options'))

    meta['train_info']['train_loss'].append(train_loss / (iter + 1))
    meta['train_info']['train_acc'].append(eval_results)


    if SKLEARN_AVAILABLE:
        detailed_metrics = calculate_detailed_metrics(torch.cat(pred_list), torch.cat(target_list))
        detailed_metrics['avg_loss'] = train_loss / (iter + 1)


        if 'train_detailed_metrics' not in meta:
            meta['train_detailed_metrics'] = []
        meta['train_detailed_metrics'].append(detailed_metrics)


    TITLE = 'Train Results'
    TABLE_DATA = (
    ('Top-1 Acc', 'Top-5 Acc', 'Mean Precision', 'Mean Recall', 'Mean F1 Score'),
    ('{:.2f}'.format(eval_results.get('accuracy_top-1', 0.0)),
     '{:.2f}'.format(eval_results.get('accuracy_top-5', 100.0)),
     '{:.2f}'.format(mean(eval_results.get('precision', 0.0))),
     '{:.2f}'.format(mean(eval_results.get('recall', 0.0))),
     '{:.2f}'.format(mean(eval_results.get('f1_score', 0.0)))),
    )
    table_instance = AsciiTable(TABLE_DATA, TITLE)
    print()
    print(table_instance.table)
    print()
