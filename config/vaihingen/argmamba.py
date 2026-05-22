from torch.utils.data import DataLoader
from geoseg.losses import *
from geoseg.datasets.vaihingen_dataset import *
from geoseg.models.argmamba import ARGMamba
from catalyst.contrib.nn import Lookahead
from catalyst import utils
import torch
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
import math

DATA_ROOT = '/data1/jzl/RemoteSensing-Seg-master/RemoteSensing-Seg-master/data/vaihingen'

max_epoch = 200
ignore_index = len(CLASSES)
train_batch_size = 1
val_batch_size = 1
lr = 1e-4
weight_decay = 0.01
backbone_lr = 6e-5
backbone_weight_decay = 0.01
num_classes = len(CLASSES)
classes = CLASSES

weights_name = 'ARGMamba-rgbd'
weights_path = 'model_weights/vaihingen/{}'.format(weights_name)
test_weights_name = 'ARGMamba-rgbd'
log_name = 'vaihingen/{}'.format(weights_name)
monitor = 'val_mIoU'
monitor_mode = 'max'
save_top_k = 1
save_last = True
check_val_every_n_epoch = 1
pretrained_ckpt_path = None
gpus = 1
resume_ckpt_path = None
deep_supervision = True

aux_tasks = True
lambda_edge = 0.5
lambda_depth = 0.2
use_dsm = True

net = ARGMamba(
    num_classes=num_classes,
    deep_supervision=deep_supervision,
    aux_tasks=aux_tasks,
    boundary_refine=True,
    use_context_module=True,
    window_sizes=(8, 16),
)

loss = JointLoss(
    SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
    DiceLoss(smooth=0.05, ignore_index=ignore_index),
    1.0,
    1.0,
)

edge_loss_fn = nn.BCEWithLogitsLoss()
depth_loss_fn = nn.MSELoss()


def get_training_transform():
    train_transform = [albu.Normalize()]
    return albu.Compose(train_transform, additional_targets={'dsm': 'image'})


def train_aug(img, mask, dsm=None):
    crop_aug = Compose([
        RandomScale(scale_list=[0.5, 0.75, 1.0, 1.25, 1.5], mode='value'),
        SmartCropV1(crop_size=512, max_ratio=0.75, ignore_index=len(CLASSES), nopad=False),
    ])

    if dsm is None:
        img, mask = crop_aug(img, mask)
        img, mask = np.array(img), np.array(mask)
        aug = get_training_transform()(image=img.copy(), mask=mask.copy())
        return aug['image'], aug['mask']

    img, mask, dsm = crop_aug(img, mask, dsm)
    img, mask, dsm = np.array(img), np.array(mask), np.array(dsm)
    aug = get_training_transform()(image=img.copy(), mask=mask.copy(), dsm=dsm.copy())
    return aug['image'], aug['mask'], aug['dsm']


def get_val_transform():
    val_transform = [albu.Normalize()]
    return albu.Compose(val_transform, additional_targets={'dsm': 'image'})


def val_aug(img, mask, dsm=None):
    img, mask = np.array(img), np.array(mask)
    if dsm is None:
        aug = get_val_transform()(image=img.copy(), mask=mask.copy())
        return aug['image'], aug['mask']

    dsm = np.array(dsm)
    aug = get_val_transform()(image=img.copy(), mask=mask.copy(), dsm=dsm.copy())
    return aug['image'], aug['mask'], aug['dsm']


train_dataset = VaihingenDataset(
    data_root=f'{DATA_ROOT}/train',
    mode='train',
    mosaic_ratio=0.25,
    transform=train_aug,
    use_dsm=use_dsm,
)

val_dataset = VaihingenDataset(
    data_root=f'{DATA_ROOT}/test',
    mode='val',
    transform=val_aug,
    use_dsm=use_dsm,
)

test_dataset = VaihingenDataset(
    data_root=f'{DATA_ROOT}/test',
    transform=val_aug,
    use_dsm=use_dsm,
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=train_batch_size,
    num_workers=4,
    pin_memory=True,
    shuffle=True,
    drop_last=True,
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=val_batch_size,
    num_workers=4,
    shuffle=False,
    pin_memory=True,
    drop_last=False,
)

layerwise_params = {'backbone.*': dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = utils.process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=len(train_loader), T_mult=1, verbose=True)
