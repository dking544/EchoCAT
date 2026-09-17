import torch
import torch.nn as nn


from .cross_entropy_loss import CrossEntropyLoss
from .utils import convert_to_one_hot


class LabelSmoothLoss(nn.Module):


    def __init__(self,
                 label_smooth_val,
                 num_classes=None,
                 mode='original',
                 reduction='mean',
                 loss_weight=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.loss_weight = loss_weight

        assert (isinstance(label_smooth_val, float)
                and 0 <= label_smooth_val < 1), \
            f'LabelSmoothLoss accepts a float label_smooth_val ' \
            f'over [0, 1), but gets {label_smooth_val}'
        self.label_smooth_val = label_smooth_val

        accept_reduction = {'none', 'mean', 'sum'}
        assert reduction in accept_reduction, \
            f'LabelSmoothLoss supports reduction {accept_reduction}, ' \
            f'but gets {mode}.'
        self.reduction = reduction

        accept_mode = {'original', 'classy_vision', 'multi_label'}
        assert mode in accept_mode, \
            f'LabelSmoothLoss supports mode {accept_mode}, but gets {mode}.'
        self.mode = mode

        self._eps = label_smooth_val
        if mode == 'classy_vision':
            self._eps = label_smooth_val / (1 + label_smooth_val)
        if mode == 'multi_label':
            self.ce = CrossEntropyLoss(use_sigmoid=True)
            self.smooth_label = self.multilabel_smooth_label
        else:
            self.ce = CrossEntropyLoss(use_soft=True)
            self.smooth_label = self.original_smooth_label

    def generate_one_hot_like_label(self, label):


        if label.dim() == 1 or (label.dim() == 2 and label.shape[1] == 1):
            label = convert_to_one_hot(label.view(-1, 1), self.num_classes)
        return label.float()

    def original_smooth_label(self, one_hot_like_label):
        assert self.num_classes > 0
        smooth_label = one_hot_like_label * (1 - self._eps)
        smooth_label += self._eps / self.num_classes
        return smooth_label

    def multilabel_smooth_label(self, one_hot_like_label):
        assert self.num_classes > 0
        smooth_label = torch.full_like(one_hot_like_label, self._eps)
        smooth_label.masked_fill_(one_hot_like_label > 0, 1 - self._eps)
        return smooth_label

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):


        if self.num_classes is not None:
            assert self.num_classes == cls_score.shape[1], \
                f'num_classes should equal to cls_score.shape[1], ' \
                f'but got num_classes: {self.num_classes} and ' \
                f'cls_score.shape[1]: {cls_score.shape[1]}'
        else:
            self.num_classes = cls_score.shape[1]

        one_hot_like_label = self.generate_one_hot_like_label(label=label)
        assert one_hot_like_label.shape == cls_score.shape, \
            f'LabelSmoothLoss requires output and target ' \
            f'to be same shape, but got output.shape: {cls_score.shape} ' \
            f'and target.shape: {one_hot_like_label.shape}'

        smoothed_label = self.smooth_label(one_hot_like_label)
        return self.ce.forward(
            cls_score,
            smoothed_label,
            weight=weight,
            avg_factor=avg_factor,
            reduction_override=reduction_override,
            **kwargs)
