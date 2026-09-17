import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from core.optimizers import CosineAnnealingLrUpdater
from utils.dataloader import Mydataset, collate
import utils.train_utils as legacy
from utils.train_consistency_20260823 import configure_consistency, train as paired_train
from echocat.runtime import recipe, build, CLASSES, load_lines
from echocat.metrics import training_metrics, validate, selection_key
from echocat.monitoring import EarlyStopMonitor
from echocat.checkpoints import EpochCheckpoints


def main():
    p = argparse.ArgumentParser(description='One training entry point; retain the best validation and final-epoch checkpoints.')
    p.add_argument('--task', required=True, choices=CLASSES)
    p.add_argument('--train-list', required=True)
    p.add_argument('--val-list', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed', type=int, default=3407)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--workers', type=int, default=None)
    p.add_argument('--batch-size', type=int, default=None, help='Override changes the historical recipe')
    p.add_argument('--teacher', help='Custom frozen teacher; omitted uses task-specific bundled teacher')
    p.add_argument('--selection', choices=['recipe','accuracy','macro_f1'], default='recipe')
    p.add_argument('--resume-best', help='Resume optimizer/RNG from the selected best epoch checkpoint')
    p.add_argument('--early-stop-monitor', choices=EarlyStopMonitor.SETTINGS,
                   help='Optional: binary three-epoch mean precision, or six-class fetal Macro-F1')
    p.add_argument('--val-case-map', help='JSONL image-to-fetus mapping for fetal_macro_f1')
    args = p.parse_args()
    if args.epochs < 1:
        p.error('--epochs must be positive')
    monitor, case_map = None, None
    if args.early_stop_monitor:
        if args.selection != 'recipe':
            p.error('--early-stop-monitor cannot be combined with --selection overrides')
        try:
            monitor = EarlyStopMonitor(args.early_stop_monitor, args.task)
        except ValueError as exc:
            p.error(str(exc))
        if args.epochs < monitor.window:
            p.error('Total epochs must cover the monitor window')
    if bool(args.val_case_map) != (args.early_stop_monitor == 'fetal_macro_f1'):
        p.error('--val-case-map is required only for fetal_macro_f1')
    cfg = recipe(args.task)
    batch_size = args.batch_size or cfg['data']['batch_size']
    workers = cfg['data']['num_workers'] if args.workers is None else args.workers
    if batch_size < 2 or workers < 0:
        p.error('Batch size must be >=2; workers must be >=0')
    train_lines = load_lines(args.train_list, len(CLASSES[args.task]))
    val_lines = load_lines(args.val_list, len(CLASSES[args.task]))
    if args.val_case_map:
        from echocat.fetal_validation import load_case_map, validate_fetal
        case_map = load_case_map(args.val_case_map, val_lines)
    paths = lambda lines: {x.rsplit(' ',1)[0] for x in lines}
    if paths(train_lines) & paths(val_lines):
        raise ValueError('Training and validation images overlap')
    if len(train_lines) < batch_size:
        raise ValueError('Training set smaller than batch size; no full batches')
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    checkpoints = EpochCheckpoints(out,args.resume_best)

    legacy.set_random_seed(args.seed, deterministic=True)
    np.random.default_rng(args.seed).shuffle(train_lines)
    model = build(args.task, initialize=True, teacher=args.teacher)
    opt_cfg = copy.deepcopy(cfg['optimizer']); assert opt_cfg.pop('type') == 'AdamW'
    optimizer = torch.optim.AdamW((v for v in model.parameters() if v.requires_grad), **opt_cfg)
    lr_cfg = copy.deepcopy(cfg['lr']); assert lr_cfg.pop('type') == 'CosineAnnealingLrUpdater'
    scheduler = CosineAnnealingLrUpdater(**lr_cfg)
    train_loader = DataLoader(Mydataset(train_lines, cfg['train_pipeline']),
        batch_size=batch_size, num_workers=workers, shuffle=True, pin_memory=True,
        drop_last=True, collate_fn=collate)
    val_loader = DataLoader(Mydataset(val_lines, cfg['val_pipeline']),
        batch_size=batch_size, num_workers=workers, shuffle=False, pin_memory=True,
        drop_last=False, collate_fn=collate)
    device = torch.device(args.device)
    model.to(device)
    runner = dict(optimizer=optimizer, train_loader=train_loader, val_loader=val_loader,
                  iter=0, epoch=0, max_epochs=args.epochs,
                  max_iters=args.epochs*len(train_loader), batch_mix=None)
    meta = {'train_info': {'train_loss':[], 'train_acc':[]}, 'save_dir':str(out)}
    legacy.calculate_detailed_metrics = training_metrics

    import utils.train_consistency_20260823 as paired
    paired.calculate_detailed_metrics = training_metrics
    if args.task == 'view':
        configure_consistency(cfg['data']['train']['paired_consistency'])
    train_epoch = paired_train if args.task == 'view' else legacy.train
    fingerprint = {'task':args.task, 'seed':args.seed, 'epochs':args.epochs,
        'batch_size':batch_size, 'workers':workers, 'selection':args.selection,
        'train_sha256':hashlib.sha256(Path(args.train_list).read_bytes()).hexdigest(),
        'val_sha256':hashlib.sha256(Path(args.val_list).read_bytes()).hexdigest(),
        'teacher_sha256':hashlib.sha256(Path(args.teacher or cfg['model'].get('cam_generator',{}).get('checkpoint')).read_bytes()).hexdigest() if args.task!='view' else None}
    best_key, start = None, 0
    if monitor:
        fingerprint['early_stop'] = {'monitor': monitor.name, 'patience': monitor.patience,
            'window': monitor.window, 'min_delta': 0,
            'val_case_map_sha256': hashlib.sha256(Path(args.val_case_map).read_bytes()).hexdigest() if args.val_case_map else None,
            'protocol': 'fetus_mean_probability_legacy_extra_softmax' if case_map is not None else 'image_validation'}
    scheduler.before_run(runner)
    if args.resume_best:
        saved = torch.load(args.resume_best, map_location=device, weights_only=True)
        if saved.get('checkpoint_kind') == 'last':
            raise ValueError('--resume-best requires the selected best checkpoint, not a final-only checkpoint')
        if saved['fingerprint'] != fingerprint:
            raise ValueError('Resume settings/manifests differ from the saved training run')
        model.load_state_dict(saved['state_dict'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        runner['iter'] = saved['iteration']; start = saved['epoch']
        best_key = tuple(saved['selection_key'])
        if monitor:
            monitor.load_state_dict(saved['monitor_state'])
            if monitor.epoch != start:
                raise ValueError('Checkpoint epoch and monitor state disagree')
        if checkpoints.best_path is None:
            checkpoints.save(saved,best=True,last=start==args.epochs)
        random.setstate(saved['rng']['python'])
        ns=saved['rng']['numpy']; np.random.set_state((ns[0],np.asarray(ns[1],dtype=np.uint32),ns[2],ns[3],ns[4]))
        torch.set_rng_state(saved['rng']['torch'].cpu())
        if device.type == 'cuda':torch.cuda.set_rng_state_all([x.cpu() for x in saved['rng']['cuda']])
    (out/'run_config.json').write_text(json.dumps(fingerprint,indent=2))
    for epoch in range(start, args.epochs):
        scheduler.before_train_epoch(runner)
        train_epoch(model,runner,scheduler,device,epoch,args.epochs,cfg['data']['test'],meta)
        result = validate_fetal(model,val_loader,device,case_map) if case_map is not None else validate(model,val_loader,device)
        monitored = None
        if monitor:
            if any(n == 0 for n in result['support_per_class']):
                raise ValueError('Validation set must contain every class')
            value = result['fetal']['f1_macro'] if case_map is not None else result['precision_macro']
            monitored = monitor.update(value, epoch+1)
            key = (monitored['score'],) if monitored['score'] is not None else ()
            improved = monitored['is_best']
        else:
            key = selection_key(result,args.task,args.selection)
            improved = best_key is None or key > best_key
        record = dict(epoch=epoch+1,validation=result,selection_key=list(key),is_best=improved)
        if monitored is not None:
            record['early_stop'] = monitored
        with (out/'history.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(record)+'\n')
        is_last = epoch+1 == args.epochs or (monitored is not None and monitored['should_stop'])
        if improved:
            best_key = key
        if improved or is_last:
            ns=np.random.get_state()
            rng={'python':random.getstate(),'numpy':(ns[0],ns[1].tolist(),ns[2],ns[3],ns[4]),
                 'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all() if device.type=='cuda' else []}
            checkpoint={'state_dict':model.state_dict(),'optimizer':optimizer.state_dict(),
                'epoch':epoch+1,'iteration':runner['iter'],'selection_key':list(key),
                'validation':result,'fingerprint':fingerprint,'rng':rng}
            if monitor:
                checkpoint['monitor_state'] = monitor.state_dict()
        if improved or is_last:
            if is_last and not improved:
                checkpoint = dict(checkpoint,checkpoint_kind='last',best_selection_key=list(best_key))
            checkpoints.save(checkpoint,best=improved,last=is_last)
        if improved:
            (out/'best_validation.json').write_text(json.dumps(record,indent=2))
        if is_last:
            (out/'last_validation.json').write_text(json.dumps(record,indent=2))
        print(json.dumps({'epoch':epoch+1,'val_accuracy':result['accuracy'],
                          'val_macro_f1':result['f1_macro'],'saved_best':improved}),flush=True)
        if monitored is not None and monitored['should_stop']:
            break
    if monitor:
        saved = torch.load(checkpoints.best_path, map_location=device, weights_only=True)
        model.load_state_dict(saved['state_dict'], strict=True)
        summary = {'monitor': monitor.name, 'last_epoch': monitor.epoch,
                   'restored_epoch': saved['epoch'], 'best_score': saved['monitor_state']['best_score'],
                   'early_stopped': monitor.wait >= monitor.patience}
        (out/'early_stop_summary.json').write_text(json.dumps(summary,indent=2))
        print(json.dumps(summary),flush=True)
    print('Training complete. Test data were not loaded. See checkpoints.json for retained epochs.')


if __name__ == '__main__':
    main()
