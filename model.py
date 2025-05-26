from typing import Any, Dict

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from capybara import get_curdir
from chameleon import ArcFace, CosFace
from otter import BaseMixin
from torchmetrics import Accuracy

from .component import *

DIR = get_curdir(__file__)


class CIFAR100ModelBaseline(BaseMixin, L.LightningModule):

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()

        self.cfg = cfg
        self.preview_batch = cfg.common.preview_batch
        self.apply_solver_config(cfg.optimizer, cfg.lr_scheduler)

        # Setup model
        cfg_model = cfg['model']
        self.backbone = nn.Identity()
        self.head = nn.Identity()

        if hasattr(cfg_model, 'backbone'):
            self.backbone = globals()[cfg_model.backbone.name](
                **cfg_model.backbone.options)

        if hasattr(cfg_model, 'head'):
            if hasattr(self.backbone, 'channels'):
                in_channels_list = self.backbone.channels
            else:
                in_channels_list = []

            cfg_model.head.options.update({
                'in_channels_list': in_channels_list,
            })
            self.head = globals()[cfg_model.head.name](
                **cfg_model.head.options)

        # Setup loss function
        self.loss_fcn = nn.CrossEntropyLoss()
        self.acc = Accuracy(
            task='multiclass',
            num_classes=cfg_model.head.options.num_classes
        )

        # for validation
        self.validation_step_outputs = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.head(x)
        return x

    def training_step(self, batch, batch_idx):
        imgs, gts = batch
        logits = self.forward(imgs)
        loss = self.loss_fcn(logits, gts)
        acc = self.acc(logits, gts)

        self.log_dict(
            {
                'lr': self.get_lr(),
                'loss': loss,
                'acc': acc,
            },
            prog_bar=True,
            on_step=True,
            sync_dist=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        imgs, gts = batch
        logits = self.forward(imgs)
        self.validation_step_outputs.append([logits, gts])

    def on_validation_epoch_end(self):
        preds, gts = [], []
        for pred, gt in self.validation_step_outputs:
            preds.extend(pred)
            gts.extend(gt)

        preds = torch.stack(preds, dim=0)
        gts = torch.stack(gts, dim=0)

        test_acc = self.acc(preds, gts)
        self.log('test_acc', test_acc, sync_dist=True)
        self.validation_step_outputs.clear()


class CIFAR100ModelBaselineSmooth(BaseMixin, L.LightningModule):

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()

        self.cfg = cfg
        self.preview_batch = cfg.common.preview_batch
        self.apply_solver_config(cfg.optimizer, cfg.lr_scheduler)

        # Setup model
        cfg_model = cfg['model']
        self.backbone = nn.Identity()
        self.head = nn.Identity()

        if hasattr(cfg_model, 'backbone'):
            self.backbone = globals()[cfg_model.backbone.name](
                **cfg_model.backbone.options)

        if hasattr(cfg_model, 'head'):
            if hasattr(self.backbone, 'channels'):
                in_channels_list = self.backbone.channels
            else:
                in_channels_list = []

            cfg_model.head.options.update({
                'in_channels_list': in_channels_list,
            })
            self.head = globals()[cfg_model.head.name](
                **cfg_model.head.options)

        # Setup loss function
        self.loss_fcn = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.acc = Accuracy(
            task='multiclass',
            num_classes=cfg_model.head.options.num_classes
        )

        # for validation
        self.validation_step_outputs = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.head(x)
        return x

    def training_step(self, batch, batch_idx):
        imgs, gts = batch
        logits = self.forward(imgs)
        loss = self.loss_fcn(logits, gts)
        acc = self.acc(logits, gts)

        self.log_dict(
            {
                'lr': self.get_lr(),
                'loss': loss,
                'acc': acc,
            },
            prog_bar=True,
            on_step=True,
            sync_dist=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        imgs, gts = batch
        logits = self.forward(imgs)
        self.validation_step_outputs.append([logits, gts])

    def on_validation_epoch_end(self):
        preds, gts = [], []
        for pred, gt in self.validation_step_outputs:
            preds.extend(pred)
            gts.extend(gt)

        preds = torch.stack(preds, dim=0)
        gts = torch.stack(gts, dim=0)

        test_acc = self.acc(preds, gts)
        self.log('test_acc', test_acc, sync_dist=True)
        self.validation_step_outputs.clear()


class CIFAR100ModelMargin(BaseMixin, L.LightningModule):

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()

        self.cfg = cfg
        self.preview_batch = cfg.common.preview_batch
        self.apply_solver_config(cfg.optimizer, cfg.lr_scheduler)

        # Setup model
        cfg_model = cfg['model']
        self.backbone = nn.Identity()
        self.head = nn.Identity()

        if hasattr(cfg_model, 'backbone'):
            self.backbone = globals()[cfg_model.backbone.name](
                **cfg_model.backbone.options)

        if hasattr(cfg_model, 'head'):
            if hasattr(self.backbone, 'channels'):
                in_channels_list = self.backbone.channels
            else:
                in_channels_list = []

            cfg_model.head.options.update({
                'in_channels_list': in_channels_list,
            })
            self.head = globals()[cfg_model.head.name](
                **cfg_model.head.options)

        # Setup loss function
        self.margin_softmax = ArcFace(s=8, m=0.35)
        self.loss_fcn = nn.CrossEntropyLoss()
        self.acc = Accuracy(
            task='multiclass',
            num_classes=cfg_model.head.options.num_classes
        )

        # for validation
        self.validation_step_outputs = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.head(x)
        return x

    def training_step(self, batch, batch_idx):
        imgs, gts = batch
        logits = self.forward(imgs)
        logits = self.margin_softmax(logits, gts)
        loss = self.loss_fcn(logits, gts)
        acc = self.acc(logits, gts)

        self.log_dict(
            {
                'lr': self.get_lr(),
                'loss': loss,
                'acc': acc,
            },
            prog_bar=True,
            on_step=True,
            sync_dist=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        imgs, gts = batch
        logits = self.forward(imgs)
        self.validation_step_outputs.append([logits, gts])

    def on_validation_epoch_end(self):
        preds, gts = [], []
        for pred, gt in self.validation_step_outputs:
            preds.extend(pred)
            gts.extend(gt)

        preds = torch.stack(preds, dim=0)
        gts = torch.stack(gts, dim=0)

        test_acc = self.acc(preds, gts)
        self.log('test_acc', test_acc, sync_dist=True)
        self.validation_step_outputs.clear()


class CIFAR100ModelMarginKD(BaseMixin, L.LightningModule):

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()

        self.cfg = cfg
        self.preview_batch = cfg.common.preview_batch
        self.apply_solver_config(cfg.optimizer, cfg.lr_scheduler)

        # Setup model
        cfg_model = cfg['model']
        self.backbone = nn.Identity()
        self.head = nn.Identity()

        if hasattr(cfg_model, 'backbone'):
            self.backbone = globals()[cfg_model.backbone.name](
                **cfg_model.backbone.options)

        if hasattr(cfg_model, 'head'):
            if hasattr(self.backbone, 'channels'):
                in_channels_list = self.backbone.channels
            else:
                in_channels_list = []

            cfg_model.head.options.update({
                'in_channels_list': in_channels_list,
            })
            self.head = globals()[cfg_model.head.name](
                **cfg_model.head.options)

        # Stop training for teacher model
        self.head.requires_grad_(False)
        self.backbone.requires_grad_(False)

        # Setup student model
        self.student = Backbone(
            name="timm_resnet18",
            pretrained=True,
            features_only=True
        )
        self.student_head = MarginHead(
            in_channels_list=self.student.channels,
            hid_dim=cfg_model.head.options.hid_dim,
            num_classes=cfg_model.head.options.num_classes
        )



        # Setup loss function
        self.margin_softmax = ArcFace(s=8, m=0.35)
        self.loss_fcn = nn.CrossEntropyLoss()
        self.loss_kd = nn.KLDivLoss(reduction='batchmean')
        self.loss_mse = nn.MSELoss()
        self.acc = Accuracy(
            task='multiclass',
            num_classes=cfg_model.head.options.num_classes
        )

        # for validation
        self.validation_step_outputs = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.head(x)
        return x

    def training_step(self, batch, batch_idx):
        imgs, large_imgs, gts = batch

        # student forward
        logits_student = self.student_head(self.student(imgs))
        logits_student = self.margin_softmax(logits_student, gts)
        ce_loss = self.loss_fcn(logits_student, gts)
        acc = self.acc(logits_student, gts)

        # teacher forward
        with torch.no_grad():
            logits_teacher = self.forward(large_imgs)
            logits_teacher = self.margin_softmax(logits_teacher, gts)

        loss_kd = self.loss_kd(
            F.log_softmax(logits_student / 4, dim=-1),
            F.softmax(logits_teacher / 4, dim=-1)
        ) * 16

        loss = 0.7 * loss_kd + 0.3 * ce_loss

        self.log_dict(
            {
                'lr': self.get_lr(),
                'loss': loss,
                'loss_kd': loss_kd,
                'loss_ce': ce_loss,
                'acc': acc,
            },
            prog_bar=True,
            on_step=True,
            sync_dist=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        imgs, large_imgs, gts = batch
        logits = self.student_head(self.student(imgs))
        self.validation_step_outputs.append([logits, gts])

    def on_validation_epoch_end(self):
        preds, gts = [], []
        for pred, gt in self.validation_step_outputs:
            preds.extend(pred)
            gts.extend(gt)

        preds = torch.stack(preds, dim=0)
        gts = torch.stack(gts, dim=0)

        test_acc = self.acc(preds, gts)
        self.log('test_acc', test_acc, sync_dist=True)
        self.validation_step_outputs.clear()


class CIFAR100Model(BaseMixin, L.LightningModule):

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()

        self.cfg = cfg
        self.preview_batch = cfg.common.preview_batch
        self.apply_solver_config(cfg.optimizer, cfg.lr_scheduler)

        # Setup model
        cfg_model = cfg['model']
        self.backbone = nn.Identity()
        self.head = nn.Identity()

        if hasattr(cfg_model, 'backbone'):
            self.backbone = globals()[cfg_model.backbone.name](
                **cfg_model.backbone.options)

        if hasattr(cfg_model, 'head'):
            if hasattr(self.backbone, 'channels'):
                in_channels_list = self.backbone.channels
            else:
                in_channels_list = []

            cfg_model.head.options.update({
                'in_channels_list': in_channels_list,
            })
            self.head = globals()[cfg_model.head.name](
                **cfg_model.head.options)

        # clip model
        # self.clip_model, _ = clip.load('ViT-B/32', device='cuda')
        # self.clip_model.requires_grad_(False)
        # self.proj = nn.Sequential(
        #     nn.Linear(512, 512, bias=False),
        #     nn.BatchNorm1d(512),
        #     nn.Linear(512, 512, bias=False),
        #     nn.BatchNorm1d(512),
        #     nn.Linear(512, 100),
        # )

        # Setup loss function
        self.margin_softmax = CosFace(s=8, m=0.3)
        self.loss_fcn = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.loss_kl = nn.KLDivLoss(reduction='batchmean', log_target=True)
        self.acc = Accuracy(
            task='multiclass',
            num_classes=cfg_model.head.options.num_classes
        )

        # for validation
        self.validation_step_outputs = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.head(x)
        return x

    def training_step(self, batch, batch_idx):
        imgs, clip_imgs, gts = batch
        logits, norm_embeddings = self.forward(imgs)
        logits = self.margin_softmax(logits, gts)

        # clip_feats = self.clip_model.encode_image(clip_imgs)
        # clip_feats = normalize(clip_feats, dim=-1)
        # logits = self.proj(clip_feats.float())
        # clip_feats = clip_feats.log_softmax(dim=-1).detach()


        # norm_embeddings = norm_embeddings.log_softmax(dim=-1)

        # loss_kl = self.loss_kl(norm_embeddings, clip_feats)
        loss = self.loss_fcn(logits, gts)

        # loss = loss_ce + loss_kl
        acc = self.acc(logits, gts)

        self.log_dict(
            {
                'lr': self.get_lr(),
                'loss': loss,
                # 'loss_kl': loss_kl,
                # 'loss_ce': loss_ce,
                'acc': acc,
            },
            prog_bar=True,
            on_step=True,
            sync_dist=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        imgs, clip_imgs, gts = batch
        x = self.backbone(imgs)
        logits, _ = self.head(x)

        # clip_feats = self.clip_model.encode_image(clip_imgs)
        # clip_feats = normalize(clip_feats, dim=-1)
        # clip_feats = clip_feats.log_softmax(dim=-1).detach()
        # logits = self.proj(clip_feats.float())

        self.validation_step_outputs.append([logits, gts])

    def on_validation_epoch_end(self):
        preds, gts = [], []
        for pred, gt in self.validation_step_outputs:
            preds.extend(pred)
            gts.extend(gt)

        preds = torch.stack(preds, dim=0)
        gts = torch.stack(gts, dim=0)

        test_acc = self.acc(preds, gts)
        self.log('test_acc', test_acc, sync_dist=True)
        self.validation_step_outputs.clear()
