from configs.backbones import *
from configs.necks import *
from configs.heads import *
from configs.common import BaseModule,Sequential

import torch.nn as nn
import torch.nn.functional as F
import torch

import functools
from inspect import getfullargspec
from collections import abc
import numpy as np

from models.paper_branch import PaperResidualBranch
from models.cnn_cam import FrozenResNet18CamGenerator
from models.domain_generalization import build_feature_augmentation

def build_model(cfg):
    if isinstance(cfg, list):
        modules = [
            eval(cfg_.pop("type"))(**cfg_) for cfg_ in cfg
        ]
        return Sequential(*modules)
    else:
        return eval(cfg.pop("type"))(**cfg)


class BuildNet(BaseModule):
    def __init__(self,cfg):
        super(BuildNet, self).__init__()
        self.neck_cfg = cfg.get("neck")
        self.head_cfg = cfg.get("head")
        self.backbone = build_model(cfg.get("backbone"))
        if self.neck_cfg is not None:
            self.neck = build_model(cfg.get("neck"))

        if self.head_cfg is not None:
            self.head = build_model(cfg.get("head"))

        self.paper_branch_cfg = cfg.get("paper_branch")
        self.paper_branch = None
        if self.paper_branch_cfg is not None:
            branch_cfg = dict(self.paper_branch_cfg)
            branch_type = branch_cfg.pop("type", "PaperResidualBranch")
            if branch_type != "PaperResidualBranch":
                raise ValueError(
                    f'Unsupported paper branch type: {branch_type}')
            self.paper_branch = PaperResidualBranch(**branch_cfg)

        self.cam_generator_cfg = cfg.get("cam_generator")
        self.cam_generator = None
        if self.cam_generator_cfg is not None:
            cam_cfg = dict(self.cam_generator_cfg)
            cam_type = cam_cfg.pop("type", "FrozenResNet18CamGenerator")
            if cam_type != "FrozenResNet18CamGenerator":
                raise ValueError(f'Unsupported CAM generator type: {cam_type}')
            self.cam_generator = FrozenResNet18CamGenerator(**cam_cfg)
        if self.paper_branch is not None and self.cam_generator is None:
            raise ValueError('paper_branch requires an independent cam_generator.')


        self.feature_augmentation_cfg = cfg.get("feature_augmentation")
        self.feature_augmentation = None
        self._feature_augmentation_hook = None
        if self.feature_augmentation_cfg is not None:
            if not hasattr(self.backbone, "patch_embed"):
                raise TypeError(
                    "feature_augmentation requires backbone.patch_embed")
            self.feature_augmentation = build_feature_augmentation(
                self.feature_augmentation_cfg)

            def apply_feature_augmentation(_module, _inputs, output):
                return self.feature_augmentation(output)

            self._feature_augmentation_hook = (
                self.backbone.patch_embed.register_forward_hook(
                    apply_feature_augmentation))

    def freeze_layers(self,names):
        assert isinstance(names,tuple)
        for name in names:
            layers = getattr(self, name)

            for param in layers.parameters():
                param.requires_grad = False

    def extract_feat(self, img, stage='neck'):


        assert stage in ['backbone', 'neck', 'pre_logits'], \
            (f'Invalid output stage "{stage}", please choose from "backbone", '
             '"neck" and "pre_logits"')

        x = self.backbone(img)

        if stage == 'backbone':
            return x

        if hasattr(self, 'neck') and self.neck is not None:
            x = self.neck(x)
        if stage == 'neck':
            return x

    def forward(self, x, return_loss=True, train_statu=False, **kwargs):
        branch_diagnostics = None
        if self.paper_branch is None:
            x = self.extract_feat(x)
        else:
            images = x
            x, spatial = self.backbone(x, return_spatial=True)
            if hasattr(self, 'neck') and self.neck is not None:
                x = self.neck(x)

            if not isinstance(x, tuple) or not isinstance(spatial, tuple):
                raise TypeError(
                    'The paper branch expects tuple outputs from TinyViT.')
            base_feature = x[-1]
            targets = kwargs.get('targets')
            reconstruct = self.training and targets is not None
            num_tokens = spatial[-1].shape[1]
            side = int(np.sqrt(num_tokens))
            if side * side != num_tokens:
                raise ValueError(f'Expected square token map, got {num_tokens}.')


            if reconstruct:
                external_cam = self.cam_generator(
                    images, output_size=(side, side))
            else:
                external_cam = spatial[-1].new_zeros(
                    spatial[-1].shape[0], num_tokens)
            residual, branch_diagnostics = self.paper_branch(
                spatial_tokens=spatial[-1],
                images=images,
                external_cam=external_cam,
                reconstruct=reconstruct)
            fused = list(x)
            fused[-1] = base_feature + residual
            x = tuple(fused)

        if not train_statu:
            if return_loss:
                losses = self.forward_train(x, **kwargs)
                return self._add_paper_loss(losses, branch_diagnostics)
            else:
                return self.forward_test(x, **kwargs)
        else:
            losses = self.forward_train(x, **kwargs)
            losses = self._add_paper_loss(losses, branch_diagnostics)
            return self.forward_test(x), losses

    def _add_paper_loss(self, losses, diagnostics):
        if diagnostics is None or self.paper_branch is None:
            return losses
        classification_loss = losses['loss'].detach()
        reconstruction_loss = diagnostics['reconstruction_loss']
        weighted_loss = (
            reconstruction_loss * self.paper_branch.reconstruction_weight)
        losses['loss'] = losses['loss'] + weighted_loss

        losses['classification_loss'] = classification_loss
        losses['paper_reconstruction_loss'] = reconstruction_loss.detach()
        return losses

    def forward_train(self, x, targets, **kwargs):


        compute_loss = getattr(self.head, 'compute_loss', None)
        if getattr(compute_loss, 'use_soft', False):
            if targets.ndim == 2 and targets.shape[1] == 1:
                targets = targets.squeeze(1)
            if targets.ndim == 1:
                targets = F.one_hot(
                    targets.long(), num_classes=self.head.num_classes)
                targets = targets.to(dtype=x[-1].dtype)
            elif targets.ndim != 2 or targets.shape[1] != self.head.num_classes:
                raise ValueError(
                    'Soft-label classification expects [N] hard labels or '
                    f'[N, {self.head.num_classes}] soft labels, got '
                    f'{tuple(targets.shape)}')

        losses = dict()
        loss = self.head.forward_train(x, targets, **kwargs)
        losses.update(loss)
        return losses

    def forward_test(self, x, **kwargs):

        out = self.head.simple_test(x,**kwargs)
        return out
