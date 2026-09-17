import inspect
import copy
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.instancenorm import _InstanceNorm

from .activations import *
from .convolution import *
from .normalization import *
from .padding import *
from .drop import *
from .wrappers import *

CONV_LAYERS = ['Conv1d','Conv2d','Conv3d','Conv','Conv2dAdaptivePadding']
NORM_LAYERS = ['BN','BN1d','BN2d','BN3d','SyncBN','GN','LN','IN','IN1d','IN2d','IN3d','LN2d']

PADDING_LAYERS = ['zero', 'reflect', 'replicate']


def build_conv_layer(cfg, *args, **kwargs):


    if cfg is None:
        cfg_ = dict(type='Conv2d')
    else:
        if not isinstance(cfg, dict):
            raise TypeError('cfg must be a dict')
        if 'type' not in cfg:
            raise KeyError('the cfg dict must contain the key "type"')
        cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in CONV_LAYERS:
        raise KeyError(f'Unrecognized layer type {layer_type}')
    else:
        conv_layer = eval(layer_type)

    layer = conv_layer(*args, **kwargs, **cfg_)

    return layer

def infer_abbr(class_type):


    if not inspect.isclass(class_type):
        raise TypeError(
            f'class_type must be a type, but got {type(class_type)}')
    if hasattr(class_type, '_abbr_'):
        return class_type._abbr_
    if issubclass(class_type, _InstanceNorm):
        return 'in'
    elif issubclass(class_type, _BatchNorm):
        return 'bn'
    elif issubclass(class_type, nn.GroupNorm):
        return 'gn'
    elif issubclass(class_type, nn.LayerNorm):
        return 'ln'
    else:
        class_name = class_type.__name__.lower()
        if 'batch' in class_name:
            return 'bn'
        elif 'group' in class_name:
            return 'gn'
        elif 'layer' in class_name:
            return 'ln'
        elif 'instance' in class_name:
            return 'in'
        else:
            return 'norm_layer'

def build_norm_layer(cfg, num_features, postfix=''):


    if not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    if 'type' not in cfg:
        raise KeyError('the cfg dict must contain the key "type"')
    cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in NORM_LAYERS:
        raise KeyError(f'Unrecognized norm type {layer_type}')

    norm_layer = eval(layer_type)('')
    abbr = infer_abbr(norm_layer)

    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)

    requires_grad = cfg_.pop('requires_grad', True)
    cfg_.setdefault('eps', 1e-5)
    if layer_type != 'GN':
        layer = norm_layer(num_features, **cfg_)
        if layer_type == 'SyncBN' and hasattr(layer, '_specify_ddp_gpu_num'):
            layer._specify_ddp_gpu_num(1)
    else:
        assert 'num_groups' in cfg_
        layer = norm_layer(num_channels=num_features, **cfg_)

    for param in layer.parameters():
        param.requires_grad = requires_grad

    return name,layer

def build_activation_layer(cfg):


    cfg_ = copy.deepcopy(cfg)
    return eval(cfg_.pop('type'))(**cfg_)

def build_padding_layer(cfg, *args, **kwargs):


    if not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    if 'type' not in cfg:
        raise KeyError('the cfg dict must contain the key "type"')

    cfg_ = cfg.copy()
    padding_type = cfg_.pop('type')
    if padding_type not in PADDING_LAYERS:
        raise KeyError(f'Unrecognized padding type {padding_type}.')
    else:
        padding_layer = eval(padding_type)

    layer = padding_layer(*args, **kwargs, **cfg_)

    return layer


def build_dropout(cfg):
    cfg_ = cfg.copy()
    return eval(cfg_.pop('type'))(**cfg_)
