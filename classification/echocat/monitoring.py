import math


class EarlyStopMonitor:
    SETTINGS = {'macro_precision_3mean': ('binary', 3, 10),
                'fetal_macro_f1': ('six', 1, 15)}

    def __init__(self, name, task):
        if name not in self.SETTINGS or self.SETTINGS[name][0] != task:
            raise ValueError('Monitor is incompatible with the selected task')
        self.name = name
        _, self.window, self.patience = self.SETTINGS[name]
        self.recent = []
        self.epoch = 0
        self.best_score = None
        self.best_epoch = None
        self.wait = 0

    def update(self, value, epoch):
        value = float(value)
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('Monitor value must be finite and within [0, 1]')
        if epoch != self.epoch + 1:
            raise ValueError('Monitor epochs must be consecutive')
        self.epoch = epoch
        self.recent = (self.recent + [value])[-self.window:]
        score = sum(self.recent) / self.window if len(self.recent) == self.window else None
        improved = score is not None and (self.best_score is None or score > self.best_score)
        if improved:
            self.best_score, self.best_epoch, self.wait = score, epoch, 0
        elif score is not None:
            self.wait += 1
        return {'name': self.name, 'raw_value': value, 'score': score,
                'is_best': improved, 'best_epoch': self.best_epoch, 'wait': self.wait,
                'should_stop': self.wait >= self.patience}

    def state_dict(self):
        return {'name': self.name, 'recent': self.recent.copy(), 'epoch': self.epoch,
                'best_score': self.best_score, 'best_epoch': self.best_epoch, 'wait': self.wait}

    def load_state_dict(self, state):
        if state['name'] != self.name:
            raise ValueError('Checkpoint monitor differs from requested monitor')
        if len(state['recent']) != self.window or state['epoch'] < self.window:
            raise ValueError('Invalid monitor history in best checkpoint')
        if state['best_epoch'] != state['epoch'] or state['wait'] != 0:
            raise ValueError('Resume requires a best-monitor checkpoint')
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in state['recent']):
            raise ValueError('Invalid monitor values in checkpoint')
        if not math.isclose(state['best_score'], sum(state['recent']) / self.window, abs_tol=1e-15):
            raise ValueError('Inconsistent monitor score in checkpoint')
        self.recent = state['recent'].copy()
        self.epoch, self.best_score = state['epoch'], state['best_score']
        self.best_epoch, self.wait = state['best_epoch'], state['wait']
