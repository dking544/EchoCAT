import functools

import torch
import torch.nn.functional as F


def reduce_loss(loss, reduction):


    reduction_enum = F._Reduction.get_enum(reduction)

    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):


    if weight is not None:
        loss = loss * weight


    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:

        if reduction == 'mean':
            loss = loss.sum() / avg_factor

        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def weighted_loss(loss_func):


    @functools.wraps(loss_func)
    def wrapper(pred,
                target,
                weight=None,
                reduction='mean',
                avg_factor=None,
                **kwargs):

        loss = loss_func(pred, target, **kwargs)
        loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
        return loss

    return wrapper


def convert_to_one_hot(targets: torch.Tensor, classes) -> torch.Tensor:


    assert (torch.max(targets).item() <
            classes), 'Class Index must be less than number of classes'
    one_hot_targets = F.one_hot(
        targets.long().squeeze(-1), num_classes=classes)
    return one_hot_targets
