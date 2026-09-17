import json
import os
from pathlib import Path
import torch


class EpochCheckpoints:
    def __init__(self, output, resume=None):
        self.root = Path(output).resolve()
        self.index = self.root/'checkpoints.json'
        self.best_epoch = self.last_epoch = None
        existing = set(self.root.glob('*.pth'))
        if self.index.exists():
            if not resume:
                raise FileExistsError('Output already has checkpoints; use a new output or --resume-best')
            record = json.loads(self.index.read_text(encoding='utf-8'))
            self.best_epoch, self.last_epoch = record['best_epoch'], record['last_epoch']
            for role in ['best', 'last']:
                path = self.path(getattr(self, role+'_epoch'))
                if record[role+'_file'] != (path.name if path else None):
                    raise ValueError('Invalid checkpoint index')
                if path is not None and not path.is_file():
                    raise FileNotFoundError('Indexed checkpoint is missing')
            if Path(resume).resolve() != self.best_path:
                raise ValueError('Use a new output directory to resume a different checkpoint')
            if existing != self.paths():
                raise ValueError('Output contains untracked checkpoints; use a new output directory')
        elif existing:
            raise FileExistsError('Use an empty output directory for this training run')

    def path(self, epoch):
        if epoch is None:
            return None
        if type(epoch) is not int or epoch < 1:
            raise ValueError('Checkpoint epoch must be a positive integer')
        path = self.root/f'epoch_{epoch}.pth'
        if path.is_symlink():
            raise ValueError('Checkpoint paths cannot be symlinks')
        return path

    @property
    def best_path(self):
        return self.path(self.best_epoch)

    def paths(self):
        return {self.path(epoch) for epoch in [self.best_epoch,self.last_epoch] if epoch is not None}

    def save(self, checkpoint, best=False, last=False):
        if not (best or last):
            raise ValueError('Checkpoint must be selected as best or final')
        previous = self.paths()
        target = self.path(checkpoint['epoch'])
        if target.exists() and target not in previous:
            raise FileExistsError('Refusing to overwrite an untracked checkpoint')
        temporary = target.with_suffix('.pth.tmp')
        torch.save(checkpoint,temporary)
        os.replace(temporary,target)
        if best:
            self.best_epoch = checkpoint['epoch']
        if last:
            self.last_epoch = checkpoint['epoch']
        record = {'best_epoch':self.best_epoch,'last_epoch':self.last_epoch,
                  'best_file':self.best_path.name if self.best_path else None,
                  'last_file':self.path(self.last_epoch).name if self.last_epoch else None}
        index_tmp = self.index.with_suffix('.json.tmp')
        index_tmp.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
        os.replace(index_tmp,self.index)
        for obsolete in previous-self.paths():
            if obsolete.parent != self.root or obsolete.is_symlink():
                raise ValueError('Refusing to remove an unsafe checkpoint path')
            obsolete.unlink()
