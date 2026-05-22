import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from tools.cfg import py2cfg
from tools.metric import Evaluator
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
from torch import nn
import torch.nn.functional as F
import cv2
import numpy as np
import argparse
from pathlib import Path
import random

os.environ['CUDNN_BENCHMARK'] = 'True'


def compute_edge_gt(mask: torch.Tensor):
    mask_float = mask.unsqueeze(1).float()
    kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=mask.device).view(1, 1, 3, 3)
    kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=mask.device).view(1, 1, 3, 3)
    edge_x = F.conv2d(mask_float, kernel_x, padding=1)
    edge_y = F.conv2d(mask_float, kernel_y, padding=1)
    edge = torch.sqrt(edge_x ** 2 + edge_y ** 2)
    return (edge > 0.1).float()


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_path', type=Path, help='Path to the config.', required=True)
    return parser.parse_args()


class Supervision_Train(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.net = config.net
        self.loss = config.loss
        self.deep_supervision = config.get('deep_supervision', False)
        self.use_dsm = hasattr(config, 'use_dsm') and config.use_dsm
        self.aux_tasks = config.get('aux_tasks', False)
        if self.aux_tasks:
            self.loss_depth = nn.L1Loss()
            self.lambda_edge = config.get('lambda_edge', 0.5)
            self.lambda_depth = config.get('lambda_depth', 0.2)
            self.loss_edge = nn.BCELoss()

        self.boundary_refine = config.get('boundary_refine', False)
        if self.boundary_refine:
            self.loss_boundary = nn.BCELoss()
            self.lambda_boundary = config.get('lambda_boundary', 0.5)

        self.metrics_train = Evaluator(num_class=config.num_classes)
        self.metrics_val = Evaluator(num_class=config.num_classes)

    def forward(self, x, dsm=None):
        if self.use_dsm and dsm is not None:
            return self.net(x, dsm)
        return self.net(x)

    def training_step(self, batch, batch_idx):
        img, mask = batch['img'], batch['gt_semantic_seg']
        dsm = batch.get('dsm')
        prediction = self.forward(img, dsm)

        seg_prediction, edge_pred, depth_pred, boundary_pred = None, None, None, None
        if self.training and (self.aux_tasks or self.boundary_refine):
            if len(prediction) == 4:
                seg_prediction, edge_pred, depth_pred, boundary_pred = prediction
            elif len(prediction) == 3:
                seg_prediction, edge_pred, depth_pred = prediction
            else:
                seg_prediction = prediction[0]
        else:
            seg_prediction = prediction

        loss_seg = 0
        if self.deep_supervision and isinstance(seg_prediction, dict):
            loss_weights = {'p1': 1.0, 'p2': 0.8, 'p3': 0.6, 'p4': 0.4}
            for key, weight in loss_weights.items():
                if key in seg_prediction:
                    loss_seg += self.loss(seg_prediction[key], mask) * weight
            eval_prediction = seg_prediction.get('final', seg_prediction.get('p1'))
        else:
            loss_seg = self.loss(seg_prediction, mask)
            eval_prediction = seg_prediction

        total_loss = loss_seg
        edge_gt = None
        if (self.training and self.aux_tasks) or (self.training and self.boundary_refine):
            edge_gt = compute_edge_gt(mask)

        if self.training and self.aux_tasks:
            loss_edge = self.loss_edge(edge_pred, edge_gt)
            loss_depth = 0
            if dsm is not None:
                loss_depth = self.loss_depth(depth_pred, dsm)
            total_loss += self.lambda_edge * loss_edge + self.lambda_depth * loss_depth
            self.log('train_loss_edge', loss_edge, on_step=True, on_epoch=True)
            if dsm is not None:
                self.log('train_loss_depth', loss_depth, on_step=True, on_epoch=True)

        if self.training and self.boundary_refine and boundary_pred is not None:
            loss_boundary = self.loss_boundary(boundary_pred, edge_gt)
            total_loss += self.lambda_boundary * loss_boundary
            self.log('train_loss_boundary', loss_boundary, on_step=True, on_epoch=True)

        pre_mask = torch.argmax(eval_prediction, dim=1)
        self.metrics_train.add_batch(pre_mask.cpu().numpy(), mask.cpu().numpy())
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        return {'loss': total_loss}

    def on_train_epoch_end(self):
        if 'vaihingen' in self.config.log_name or 'potsdam' in self.config.log_name or 'whubuilding' in self.config.log_name or 'massbuilding' in self.config.log_name or 'cropland' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        else:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union())
            F1 = np.nanmean(self.metrics_train.F1())

        OA = np.nanmean(self.metrics_train.OA())
        iou_per_class = self.metrics_train.Intersection_over_Union()
        print('train:', {'mIoU': mIoU, 'F1': F1, 'OA': OA})
        print({class_name: iou for class_name, iou in zip(self.config.classes, iou_per_class)})
        self.metrics_train.reset()
        self.log_dict({'train_mIoU': mIoU, 'train_F1': F1, 'train_OA': OA}, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        img, mask = batch['img'], batch['gt_semantic_seg']
        dsm = batch.get('dsm')

        def get_final_prediction(output):
            if isinstance(output, dict):
                return output.get('final', output.get('p1'))
            return output

        pred_orig = get_final_prediction(self.forward(img, dsm))
        pred_hflip = get_final_prediction(self.forward(torch.flip(img, dims=[3]), torch.flip(dsm, dims=[3]) if dsm is not None else None))
        pred_hflip_restored = torch.flip(pred_hflip, dims=[3])
        pred_vflip = get_final_prediction(self.forward(torch.flip(img, dims=[2]), torch.flip(dsm, dims=[2]) if dsm is not None else None))
        pred_vflip_restored = torch.flip(pred_vflip, dims=[2])
        pred_hvflip = get_final_prediction(self.forward(torch.flip(img, dims=[2, 3]), torch.flip(dsm, dims=[2, 3]) if dsm is not None else None))
        pred_hvflip_restored = torch.flip(pred_hvflip, dims=[2, 3])
        prediction = (pred_orig + pred_hflip_restored + pred_vflip_restored + pred_hvflip_restored) / 4.0

        pre_mask = torch.argmax(prediction, dim=1)
        self.metrics_val.add_batch(pre_mask.cpu().numpy(), mask.cpu().numpy())
        loss_val = self.loss(prediction, mask)
        return {'loss_val': loss_val}

    def on_validation_epoch_end(self):
        if 'vaihingen' in self.config.log_name or 'potsdam' in self.config.log_name or 'whubuilding' in self.config.log_name or 'massbuilding' in self.config.log_name or 'cropland' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        else:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union())
            F1 = np.nanmean(self.metrics_val.F1())

        OA = np.nanmean(self.metrics_val.OA())
        iou_per_class = self.metrics_val.Intersection_over_Union()
        print('val:', {'mIoU': mIoU, 'F1': F1, 'OA': OA})
        print({class_name: iou for class_name, iou in zip(self.config.classes, iou_per_class)})
        self.metrics_val.reset()
        self.log_dict({'val_mIoU': mIoU, 'val_F1': F1, 'val_OA': OA}, prog_bar=True)
        self.lr_scheduler.step(mIoU)

    def configure_optimizers(self):
        optimizer = self.config.optimizer
        lr_scheduler = self.config.lr_scheduler
        self.lr_scheduler = lr_scheduler
        if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': lr_scheduler,
                    'monitor': self.config.monitor,
                },
            }
        return [optimizer], [lr_scheduler]

    def train_dataloader(self):
        return self.config.train_loader

    def val_dataloader(self):
        return self.config.val_loader


def main():
    args = get_args()
    config = py2cfg(args.config_path)
    seed_everything(42)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=config.save_top_k,
        monitor=config.monitor,
        save_last=config.save_last,
        mode=config.monitor_mode,
        dirpath=config.weights_path,
        filename=config.weights_name,
    )
    early_stop_callback = EarlyStopping(
        monitor=config.monitor,
        patience=30,
        verbose=True,
        mode=config.monitor_mode,
    )
    logger = CSVLogger('lightning_logs', name=config.log_name)

    model = Supervision_Train(config)
    if config.pretrained_ckpt_path:
        model = Supervision_Train.load_from_checkpoint(config.pretrained_ckpt_path, config=config)

    trainer = pl.Trainer(
        devices=config.gpus,
        max_epochs=config.max_epoch,
        accelerator='auto',
        check_val_every_n_epoch=config.check_val_every_n_epoch,
        callbacks=[checkpoint_callback, early_stop_callback],
        strategy='ddp_find_unused_parameters_true',
        logger=logger,
    )
    trainer.fit(model=model, ckpt_path=config.resume_ckpt_path)


if __name__ == '__main__':
    main()
