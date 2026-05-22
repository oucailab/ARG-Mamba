
import os
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import matplotlib.pyplot as plt
import albumentations as albu

import matplotlib.patches as mpatches
from PIL import Image
import random
from .transform import *
import tifffile # Add this import at the top
import albumentations.pytorch as AT # For ToTensorV2 if needed, though current setup uses manual to_tensor

CLASSES = ('ImSurf', 'Building', 'LowVeg', 'Tree', 'Car', 'Clutter')
PALETTE = [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 204, 0], [255, 0, 0]]

ORIGIN_IMG_SIZE = (1024, 1024)
INPUT_IMG_SIZE = (512, 512)
TEST_IMG_SIZE = (512, 512)


def get_training_transform():
    train_transform = [
        # --- 增强的数据增强管道 ---
        # 1. 几何变换
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.RandomRotate90(p=0.5),
        
        # 2. 颜色增强
        albu.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8),

        # 3. 结构性增强 (可选，但强烈推荐)
        # ElasticTransform 可能会稍微减慢数据加载速度，但效果很好
        albu.ElasticTransform(p=0.3, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        
        # GridMask 是一种非常有效的随机遮挡方法
        albu.GridMask(num_grid=3, p=0.3),

        # 4. 归一化 (必须放在最后)
        albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0)
        # --- 增强结束 ---
    ]
    # 'dsm' will also be processed by RandomRotate90 and Normalize.
    # If Normalize is not desired for DSM, or needs different params, this needs adjustment.
    return albu.Compose(train_transform, additional_targets={'dsm': 'image'})


def train_aug(img, mask, dsm=None):
    """支持DSM的数据增强函数"""
    # Ensure inputs are PIL images for custom transforms
    if not isinstance(img, Image.Image):
        img = Image.fromarray(np.uint8(img))
    if not isinstance(mask, Image.Image):
        mask = Image.fromarray(np.uint8(mask))
    
    if dsm is not None:
        if not isinstance(dsm, Image.Image): # Should be PIL 'F' from loader
            dsm = Image.fromarray(dsm.astype(np.float32), mode='F')
        elif dsm.mode != 'F':
            dsm = Image.fromarray(np.array(dsm).astype(np.float32), mode='F')

        # 1. Custom geometric transforms (RandomScale, SmartCropV1)
        # These should operate on PIL images and output PIL images
        crop_aug = Compose([
            RandomScale(scale_list=[0.5, 0.75, 1.0, 1.25, 1.5], mode='value'),
            SmartCropV1(crop_size=512, max_ratio=0.75, ignore_index=len(CLASSES), nopad=False)
        ])
        # Ensure inputs to crop_aug are aligned if sizes differ
        if img.size != dsm.size:
            dsm = dsm.resize(img.size, Image.BILINEAR) # BILINEAR is better for float data
        if img.size != mask.size: # Should already be aligned from loader
            mask = mask.resize(img.size, Image.NEAREST)

        img, mask, dsm = crop_aug(img, mask, dsm) # Expects PIL, returns PIL

        # Convert to NumPy for Albumentations' main geometric + normalize transform
        img_np = np.array(img)
        mask_np = np.array(mask)
        dsm_np = np.array(dsm).astype(np.float32) # dsm is float [0,1]

        if dsm_np.ndim == 3 and dsm_np.shape[-1] == 1:
            dsm_np = dsm_np.squeeze(-1)
        # dsm_np should be H, W at this point

        # 2. Albumentations geometric transforms (RandomRotate90) and Normalize (for RGB)
        # We need to ensure Normalize doesn't corrupt the pre-normalized DSM.
        # One way: apply geometric to all, then Normalize only to RGB.
        
        # Define a transform for geometric operations that applies to all
        geometric_transforms = albu.Compose([
            albu.RandomRotate90(p=0.5)
        ], additional_targets={'dsm_channel': 'image'}) # Pass dsm_np as HWC

        # Apply geometric transforms
        # dsm_np needs to be HWC for 'image' target type in albumentations
        augmented_geom = geometric_transforms(image=img_np.copy(), mask=mask_np.copy(), dsm_channel=dsm_np[..., np.newaxis].copy())
        img_geom = augmented_geom['image']
        mask_geom = augmented_geom['mask']
        dsm_geom_np = augmented_geom['dsm_channel'].squeeze() # Back to HW, still [0,1] float

        # Define Normalize only for RGB
        rgb_normalize = albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0)
        img_final_np = rgb_normalize(image=img_geom)['image']
        
        return img_final_np, mask_geom, dsm_geom_np

    else: # Original logic when DSM is not used
        crop_aug = Compose([RandomScale(scale_list=[0.5, 0.75, 1.0, 1.25, 1.5], mode='value'),
                            SmartCropV1(crop_size=512, max_ratio=0.75,
                                        ignore_index=len(CLASSES), nopad=False)])
        img, mask = crop_aug(img, mask) # Expects PIL, returns PIL
        img_np, mask_np = np.array(img), np.array(mask)
        
        # Apply RandomRotate90 and Normalize for RGB
        aug = albu.Compose([ 
            albu.RandomRotate90(p=0.5), 
            albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0)
            ])(image=img_np.copy(), mask=mask_np.copy())
        return aug['image'], aug['mask']


def get_val_transform():
    # For validation, only Normalize for RGB. DSM is already [0,1] float.
    # Geometric transforms are not usually applied in validation unless specified.
    val_transform = [
        albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0)
    ]
    # This compose will apply Normalize to 'image' (RGB)
    # If 'dsm' is passed, it would also be normalized by default. We want to avoid this.
    # So, val_aug will handle this separation.
    return albu.Compose(val_transform) # No additional_targets here, val_aug will manage


def val_aug(img, mask, dsm=None):
    """支持DSM的验证集增强函数"""
    # Inputs img, mask, dsm are expected to be PIL images from loader
    img_np = np.array(img)
    mask_np = np.array(mask)

    # Normalize RGB image
    rgb_normalize = albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0)
    img_final_np = rgb_normalize(image=img_np.copy())['image']

    if dsm is not None:
        dsm_np = np.array(dsm).astype(np.float32) # dsm is PIL 'F', convert to numpy float [0,1]
        if dsm_np.ndim == 3 and dsm_np.shape[-1] == 1:
            dsm_np = dsm_np.squeeze(-1)
        # dsm_np is already [0,1] float, no further normalization needed.
        return img_final_np, mask_np, dsm_np
    else:
        return img_final_np, mask_np


class VaihingenDataset(Dataset):
    def __init__(self, data_root='/data1/jzl/RemoteSensing-Seg-master/RemoteSensing-Seg-master/data/vaihingen/test', mode='val', img_dir='images_1024', mask_dir='masks_1024',
                 dsm_dir='dsm_1024', img_suffix='.tif', mask_suffix='.png', dsm_suffix='.tif',
                 transform=val_aug, mosaic_ratio=0.0, img_size=ORIGIN_IMG_SIZE, use_dsm=False):
        self.data_root = data_root
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.dsm_dir = dsm_dir
        self.img_suffix = img_suffix
        self.mask_suffix = mask_suffix
        self.dsm_suffix = dsm_suffix
        self.transform = transform
        self.mode = mode
        self.mosaic_ratio = mosaic_ratio
        self.img_size = img_size # This might be INPUT_IMG_SIZE or ORIGIN_IMG_SIZE
        self.use_dsm = use_dsm
        self.img_ids = self.get_img_ids(self.data_root, self.img_dir, self.mask_dir)

        
    def __getitem__(self, index):
        # --- 3. 重构 __getitem__ 以集成 ClassMix ---
        # 仅在训练模式下执行数据增强
        
        p_ratio = random.random()
        dsm_out = None # Initialize dsm_out

        if p_ratio > self.mosaic_ratio or self.mode == 'val' or self.mode == 'test':
            if self.use_dsm:
                img, mask, dsm_pil_f = self.load_img_mask_dsm(index) # dsm_pil_f is PIL 'F' mode
                if self.transform:
                    img_transformed, mask_transformed, dsm_transformed_np = self.transform(img, mask, dsm_pil_f)
                else: # Should always have a transform if training/val
                    img_transformed, mask_transformed, dsm_transformed_np = np.array(img), np.array(mask), np.array(dsm_pil_f).astype(np.float32)
                dsm_out = dsm_transformed_np
            else:
                # ... (your existing code for no DSM) ...
                img, mask = self.load_img_and_mask(index)
                if self.transform:
                    img_transformed, mask_transformed = self.transform(img, mask)
                else:
                    img_transformed, mask_transformed = np.array(img), np.array(mask)

        else: # Mosaic
            if self.use_dsm:
                img, mask, dsm_pil_f = self.load_mosaic_img_mask_dsm(index) # dsm_pil_f is PIL 'F' mode
                if self.transform:
                    img_transformed, mask_transformed, dsm_transformed_np = self.transform(img, mask, dsm_pil_f)
                else:
                    img_transformed, mask_transformed, dsm_transformed_np = np.array(img), np.array(mask), np.array(dsm_pil_f).astype(np.float32)
                dsm_out = dsm_transformed_np
            else:
                # ... (your existing code for no DSM mosaic) ...
                img, mask = self.load_mosaic_img_and_mask(index)
                if self.transform:
                    img_transformed, mask_transformed = self.transform(img, mask)
                else:
                    img_transformed, mask_transformed = np.array(img), np.array(mask)

        img_tensor = torch.from_numpy(img_transformed).permute(2, 0, 1).float()
        mask_tensor = torch.from_numpy(mask_transformed).long()
        
        results = dict(img_id=self.img_ids[index], img=img_tensor, gt_semantic_seg=mask_tensor)

        if self.use_dsm and dsm_out is not None:
            # dsm_out is already a processed numpy array (e.g., float32, [0,1]) from train_aug/val_aug
            # Ensure it's 2D HW before unsqueeze
            if dsm_out.ndim == 3 and dsm_out.shape[-1] == 1:
                 dsm_out = dsm_out.squeeze(-1)
            elif dsm_out.ndim != 2:
                 raise ValueError(f"DSM output has unexpected ndim: {dsm_out.ndim}")

            # Resize to target_size if not already done consistently in aug functions
            # It's better if aug functions output consistent sizes.
            # For now, let's assume aug functions handle sizing.
            # target_size = INPUT_IMG_SIZE # Or self.img_size if that's the target
            # if dsm_out.shape[0] != target_size[0] or dsm_out.shape[1] != target_size[1]:
            #     dsm_out = cv2.resize(dsm_out, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
            
            # DSM is already normalized to [0,1] by pre-processing and preserved by aug.
            # No need for normalize_dsm(dsm_out) here.
            
            dsm_tensor = torch.from_numpy(dsm_out.copy()).float().unsqueeze(0) # Add channel dim: C H W
            results['dsm'] = dsm_tensor
            
        return results


    def __len__(self):
        return len(self.img_ids)

    def get_img_ids(self, data_root, img_dir, mask_dir):
        img_filename_list = os.listdir(osp.join(data_root, img_dir))
        mask_filename_list = os.listdir(osp.join(data_root, mask_dir))
        assert len(img_filename_list) == len(mask_filename_list)
        img_ids = [str(id.split('.')[0]) for id in mask_filename_list]
        return img_ids

    # 新增方法：同时加载RGB、掩码和DSM
    def load_img_mask_dsm(self, index):
        img_id = self.img_ids[index]
        img_name = osp.join(self.data_root, self.img_dir, img_id + self.img_suffix)
        mask_name = osp.join(self.data_root, self.mask_dir, img_id + self.mask_suffix)
        dsm_name = osp.join(self.data_root, self.dsm_dir, img_id + self.dsm_suffix)
        
        img = Image.open(img_name).convert('RGB')
        mask = Image.open(mask_name).convert('L') # Mask is label
        
        try:
            # Load DSM as float32 numpy array using tifffile
            dsm_array = tifffile.imread(dsm_name)
            if dsm_array.ndim == 3 and dsm_array.shape[-1] == 1: # Handle (H, W, 1)
                dsm_array = dsm_array.squeeze(axis=-1)
            elif dsm_array.ndim == 3 and dsm_array.shape[-1] > 1: # Multi-channel DSM? Take first.
                 print(f"警告：DSM {dsm_name} 具有多个通道（{dsm_array.shape[-1]}）。使用第一个通道。")
                 dsm_array = dsm_array[..., 0]
            
            dsm_array = dsm_array.astype(np.float32) # Ensure float32, should be [0,1]
            dsm = Image.fromarray(dsm_array, mode='F') # Convert to PIL 'F' mode for transforms

        except FileNotFoundError:
            print(f"警告：找不到DSM文件 {dsm_name}。正在创建零DSM。")
            # Create a zero float32 DSM matching image size
            # Use self.img_size which should be the expected size of loaded images before aug.
            # Or, if img is already loaded, use img.size
            h, w = self.img_size[0], self.img_size[1] # Assuming img_size is (H,W)
            if 'img' in locals() and img is not None: # If img is already loaded
                 w, h = img.size # PIL size is (width, height)
            
            dsm_array = np.zeros((h, w), dtype=np.float32)
            dsm = Image.fromarray(dsm_array, mode='F')
        except Exception as e:
            print(f"加载DSM时出错 {dsm_name}: {e}。正在创建零DSM。")
            h, w = self.img_size[0], self.img_size[1]
            if 'img' in locals() and img is not None:
                 w, h = img.size
            dsm_array = np.zeros((h, w), dtype=np.float32)
            dsm = Image.fromarray(dsm_array, mode='F')

        # Ensure all loaded PIL images have same .size before passing to aug
        if mask.size != img.size:
            mask = mask.resize(img.size, Image.NEAREST)
        if dsm.size != img.size:
            dsm = dsm.resize(img.size, Image.NEAREST) # Or Image.BILINEAR for float DSM

        return img, mask, dsm # img (PIL RGB), mask (PIL L), dsm (PIL F)

    # 新增方法：支持DSM的镶嵌增强
    def load_mosaic_img_mask_dsm(self, index):
        indexes = [index] + [random.randint(0, len(self.img_ids) - 1) for _ in range(3)]
        img_a, mask_a, dsm_a_pil_f = self.load_img_mask_dsm(indexes[0]) # Returns PIL 'F'
        img_b, mask_b, dsm_b_pil_f = self.load_img_mask_dsm(indexes[1])
        img_c, mask_c, dsm_c_pil_f = self.load_img_mask_dsm(indexes[2])
        img_d, mask_d, dsm_d_pil_f = self.load_img_mask_dsm(indexes[3])

        img_a, mask_a, dsm_a = np.array(img_a), np.array(mask_a), np.array(dsm_a_pil_f).astype(np.float32)
        img_b, mask_b, dsm_b = np.array(img_b), np.array(mask_b), np.array(dsm_b_pil_f).astype(np.float32)
        img_c, mask_c, dsm_c = np.array(img_c), np.array(mask_c), np.array(dsm_c_pil_f).astype(np.float32)
        img_d, mask_d, dsm_d = np.array(img_d), np.array(mask_d), np.array(dsm_d_pil_f).astype(np.float32)
        
        # Ensure DSMs are 2D for RandomCrop if it expects HWC for 'image1'
        # Albumentations usually handles HW or HWC for image-like targets.
        # If dsm_a is HW (2D), it's fine. If HWC (e.g. from np.array(PIL 'F')), also fine.
        # The key is consistency. Let's assume they are HW (2D float arrays).
        if dsm_a.ndim == 3: dsm_a = dsm_a.squeeze(-1)
        if dsm_b.ndim == 3: dsm_b = dsm_b.squeeze(-1)
        if dsm_c.ndim == 3: dsm_c = dsm_c.squeeze(-1)
        if dsm_d.ndim == 3: dsm_d = dsm_d.squeeze(-1)


        h = self.img_size[0] # Target mosaic canvas size
        w = self.img_size[1]

        start_x = w // 4
        strat_y = h // 4
        offset_x = random.randint(start_x, (w - start_x))
        offset_y = random.randint(strat_y, (h - strat_y))

        crop_size_a = (offset_x, offset_y)
        crop_size_b = (w - offset_x, offset_y)
        crop_size_c = (offset_x, h - offset_y)
        crop_size_d = (w - offset_x, h - offset_y)

        # Use additional_targets for dsm
        random_crop_a_comp = albu.Compose([albu.RandomCrop(width=crop_size_a[0], height=crop_size_a[1])], additional_targets={'dsm_channel': 'image'})
        random_crop_b_comp = albu.Compose([albu.RandomCrop(width=crop_size_b[0], height=crop_size_b[1])], additional_targets={'dsm_channel': 'image'})
        random_crop_c_comp = albu.Compose([albu.RandomCrop(width=crop_size_c[0], height=crop_size_c[1])], additional_targets={'dsm_channel': 'image'})
        random_crop_d_comp = albu.Compose([albu.RandomCrop(width=crop_size_d[0], height=crop_size_d[1])], additional_targets={'dsm_channel': 'image'})

        # Albumentations expects HWC for image-like targets if they have channels.
        # Our DSMs are HW (2D). Add channel dim.
        croped_a = random_crop_a_comp(image=img_a.copy(), mask=mask_a.copy(), dsm_channel=dsm_a[...,np.newaxis].copy())
        croped_b = random_crop_b_comp(image=img_b.copy(), mask=mask_b.copy(), dsm_channel=dsm_b[...,np.newaxis].copy())
        croped_c = random_crop_c_comp(image=img_c.copy(), mask=mask_c.copy(), dsm_channel=dsm_c[...,np.newaxis].copy())
        croped_d = random_crop_d_comp(image=img_d.copy(), mask=mask_d.copy(), dsm_channel=dsm_d[...,np.newaxis].copy())

        img_crop_a, mask_crop_a, dsm_crop_a = croped_a['image'], croped_a['mask'], croped_a['dsm_channel'].squeeze()
        img_crop_b, mask_crop_b, dsm_crop_b = croped_b['image'], croped_b['mask'], croped_b['dsm_channel'].squeeze()
        img_crop_c, mask_crop_c, dsm_crop_c = croped_c['image'], croped_c['mask'], croped_c['dsm_channel'].squeeze()
        img_crop_d, mask_crop_d, dsm_crop_d = croped_d['image'], croped_d['mask'], croped_d['dsm_channel'].squeeze()
        
        # 合并图像块
        top_img = np.concatenate((img_crop_a, img_crop_b), axis=1)
        bottom_img = np.concatenate((img_crop_c, img_crop_d), axis=1)
        img_final_np = np.concatenate((top_img, bottom_img), axis=0)

        # 合并掩码块
        top_mask = np.concatenate((mask_crop_a, mask_crop_b), axis=1)
        bottom_mask = np.concatenate((mask_crop_c, mask_crop_d), axis=1)
        mask_final_np = np.concatenate((top_mask, bottom_mask), axis=0)
        
        top_dsm = np.concatenate((dsm_crop_a, dsm_crop_b), axis=1)
        bottom_dsm = np.concatenate((dsm_crop_c, dsm_crop_d), axis=1)
        dsm_final_np = np.concatenate((top_dsm, bottom_dsm), axis=0)

        mask_final_np = np.ascontiguousarray(mask_final_np)
        img_final_np = np.ascontiguousarray(img_final_np)
        dsm_final_np = np.ascontiguousarray(dsm_final_np) # Should be float32
        
        img_final_pil = Image.fromarray(img_final_np)
        mask_final_pil = Image.fromarray(mask_final_np)
        dsm_final_pil_f = Image.fromarray(dsm_final_np, mode='F') # Ensure float PIL

        return img_final_pil, mask_final_pil, dsm_final_pil_f

    # 保留原始方法以兼容不使用DSM的情况
    def load_img_and_mask(self, index):
        img_id = self.img_ids[index]
        img_name = osp.join(self.data_root, self.img_dir, img_id + self.img_suffix)
        mask_name = osp.join(self.data_root, self.mask_dir, img_id + self.mask_suffix)
        img = Image.open(img_name).convert('RGB')
        mask = Image.open(mask_name).convert('L')
        return img, mask

    # 保留原始方法以兼容不使用DSM的情况
    def load_mosaic_img_and_mask(self, index):
        indexes = [index] + [random.randint(0, len(self.img_ids) - 1) for _ in range(3)]
        img_a, mask_a = self.load_img_and_mask(indexes[0])
        img_b, mask_b = self.load_img_and_mask(indexes[1])
        img_c, mask_c = self.load_img_and_mask(indexes[2])
        img_d, mask_d = self.load_img_and_mask(indexes[3])

        img_a, mask_a = np.array(img_a), np.array(mask_a)
        img_b, mask_b = np.array(img_b), np.array(mask_b)
        img_c, mask_c = np.array(img_c), np.array(mask_c)
        img_d, mask_d = np.array(img_d), np.array(mask_d)

        h = self.img_size[0]
        w = self.img_size[1]

        start_x = w // 4
        strat_y = h // 4
        # The coordinates of the splice center
        offset_x = random.randint(start_x, (w - start_x))
        offset_y = random.randint(strat_y, (h - strat_y))

        crop_size_a = (offset_x, offset_y)
        crop_size_b = (w - offset_x, offset_y)
        crop_size_c = (offset_x, h - offset_y)
        crop_size_d = (w - offset_x, h - offset_y)

        random_crop_a = albu.RandomCrop(width=crop_size_a[0], height=crop_size_a[1])
        random_crop_b = albu.RandomCrop(width=crop_size_b[0], height=crop_size_b[1])
        random_crop_c = albu.RandomCrop(width=crop_size_c[0], height=crop_size_c[1])
        random_crop_d = albu.RandomCrop(width=crop_size_d[0], height=crop_size_d[1])

        croped_a = random_crop_a(image=img_a.copy(), mask=mask_a.copy())
        croped_b = random_crop_b(image=img_b.copy(), mask=mask_b.copy())
        croped_c = random_crop_c(image=img_c.copy(), mask=mask_c.copy())
        croped_d = random_crop_d(image=img_d.copy(), mask=mask_d.copy())

        img_crop_a, mask_crop_a = croped_a['image'], croped_a['mask']
        img_crop_b, mask_crop_b = croped_b['image'], croped_b['mask']
        img_crop_c, mask_crop_c = croped_c['image'], croped_c['mask']
        img_crop_d, mask_crop_d = croped_d['image'], croped_d['mask']

        top = np.concatenate((img_crop_a, img_crop_b), axis=1)
        bottom = np.concatenate((img_crop_c, img_crop_d), axis=1)
        img = np.concatenate((top, bottom), axis=0)

        top_mask = np.concatenate((mask_crop_a, mask_crop_b), axis=1)
        bottom_mask = np.concatenate((mask_crop_c, mask_crop_d), axis=1)
        mask = np.concatenate((top_mask, bottom_mask), axis=0)
        mask = np.ascontiguousarray(mask)
        img = np.ascontiguousarray(img)
        img = Image.fromarray(img)
        mask = Image.fromarray(mask)

        return img, mask


# 添加用于检查尺寸匹配的调试函数
def debug_dsm_sizes(dataset_path='data/vaihingen/train1gemi', num_samples=10):
    """检查DSM和RGB图像的尺寸是否匹配"""
    dataset = VaihingenDataset(
        data_root=dataset_path,
        use_dsm=True,
        transform=None  # 先不应用变换
    )
    
    print(f"数据集大小: {len(dataset)}")
    
    for i in range(min(num_samples, len(dataset))):
        try:
            img, mask, dsm = dataset.load_img_mask_dsm(i)
            print(f"样本 {i}: img={img.size}, mask={mask.size}, dsm={dsm.size}")
            
            # 验证img和dsm尺寸是否匹配
            if img.size != dsm.size:
                print(f"警告: 样本 {i} 中尺寸不匹配!")
                
                # 尝试修复
                dsm = dsm.resize(img.size, Image.NEAREST)
                print(f"  调整后: dsm={dsm.size}")
        except Exception as e:
            print(f"样本 {i} 处理错误: {e}")


# 保留原始可视化函数
def show_img_mask_seg(seg_path, img_path, mask_path, start_seg_index):
    # 原始代码保持不变
    seg_list = os.listdir(seg_path)
    seg_list = [f for f in seg_list if f.endswith('.png')]
    fig, ax = plt.subplots(2, 3, figsize=(18, 12))
    seg_list = seg_list[start_seg_index:start_seg_index+2]
    patches = [mpatches.Patch(color=np.array(PALETTE[i])/255., label=CLASSES[i]) for i in range(len(CLASSES))]
    for i in range(len(seg_list)):
        seg_id = seg_list[i]
        img_seg = cv2.imread(f'{seg_path}/{seg_id}', cv2.IMREAD_UNCHANGED)
        img_seg = img_seg.astype(np.uint8)
        img_seg = Image.fromarray(img_seg).convert('P')
        img_seg.putpalette(np.array(PALETTE, dtype=np.uint8))
        img_seg = np.array(img_seg.convert('RGB'))
        mask = cv2.imread(f'{mask_path}/{seg_id}', cv2.IMREAD_UNCHANGED)
        mask = mask.astype(np.uint8)
        mask = Image.fromarray(mask).convert('P')
        mask.putpalette(np.array(PALETTE, dtype=np.uint8))
        mask = np.array(mask.convert('RGB'))
        img_id = str(seg_id.split('.')[0])+'.tif'
        img = cv2.imread(f'{img_path}/{img_id}', cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax[i, 0].set_axis_off()
        ax[i, 0].imshow(img)
        ax[i, 0].set_title('RS IMAGE ' + img_id)
        ax[i, 1].set_axis_off()
        ax[i, 1].imshow(mask)
        ax[i, 1].set_title('Mask True ' + seg_id)
        ax[i, 2].set_axis_off()
        ax[i, 2].imshow(img_seg)
        ax[i, 2].set_title('Mask Predict ' + seg_id)
        ax[i, 2].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='large')


def show_seg(seg_path, img_path, start_seg_index):
    # 原始代码保持不变
    seg_list = os.listdir(seg_path)
    seg_list = [f for f in seg_list if f.endswith('.png')]
    fig, ax = plt.subplots(2, 2, figsize=(12, 12))
    seg_list = seg_list[start_seg_index:start_seg_index+2]
    patches = [mpatches.Patch(color=np.array(PALETTE[i])/255., label=CLASSES[i]) for i in range(len(CLASSES))]
    for i in range(len(seg_list)):
        seg_id = seg_list[i]
        img_seg = cv2.imread(f'{seg_path}/{seg_id}', cv2.IMREAD_UNCHANGED)
        img_seg = img_seg.astype(np.uint8)
        img_seg = Image.fromarray(img_seg).convert('P')
        img_seg.putpalette(np.array(PALETTE, dtype=np.uint8))
        img_seg = np.array(img_seg.convert('RGB'))
        img_id = str(seg_id.split('.')[0])+'.tif'
        img = cv2.imread(f'{img_path}/{img_id}', cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax[i, 0].set_axis_off()
        ax[i, 0].imshow(img)
        ax[i, 0].set_title('RS IMAGE '+img_id)
        ax[i, 1].set_axis_off()
        ax[i, 1].imshow(img_seg)
        ax[i, 1].set_title('Seg IMAGE '+seg_id)
        ax[i, 1].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='large')


def show_mask(img, mask, img_id):
    # 原始代码保持不变
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 12))
    patches = [mpatches.Patch(color=np.array(PALETTE[i])/255., label=CLASSES[i]) for i in range(len(CLASSES))]
    mask = mask.astype(np.uint8)
    mask = Image.fromarray(mask).convert('P')
    mask.putpalette(np.array(PALETTE, dtype=np.uint8))
    mask = np.array(mask.convert('RGB'))
    ax1.imshow(img)
    ax1.set_title('RS IMAGE ' + str(img_id)+'.tif')
    ax2.imshow(mask)
    ax2.set_title('Mask ' + str(img_id)+'.png')
    ax2.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='large')


# 添加新的可视化函数，显示图像、掩码和DSM
def show_img_mask_dsm(img, mask, dsm, img_id):
    """显示图像、掩码和DSM的可视化函数"""
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
    
    # 图例
    patches = [mpatches.Patch(color=np.array(PALETTE[i])/255., label=CLASSES[i]) for i in range(len(CLASSES))]
    
    # 处理掩码
    mask = mask.astype(np.uint8)
    mask_vis = Image.fromarray(mask).convert('P')
    mask_vis.putpalette(np.array(PALETTE, dtype=np.uint8))
    mask_vis = np.array(mask_vis.convert('RGB'))
    
    # 显示图像
    ax1.imshow(img)
    ax1.set_title(f'RGB Image: {img_id}.tif')
    ax1.axis('off')
    
    # 显示掩码
    ax2.imshow(mask_vis)
    ax2.set_title(f'Mask: {img_id}.png')
    ax2.axis('off')
    ax2.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='small')
    
    # 显示DSM（使用色彩映射以便于查看高度差异）
    dsm_vis = ax3.imshow(dsm, cmap='terrain')
    ax3.set_title(f'DSM: {img_id}.tif')
    ax3.axis('off')
    
    # 添加DSM色条
    cbar = fig.colorbar(dsm_vis, ax=ax3, shrink=0.6)
    cbar.set_label('高度值')
    
    plt.tight_layout()
    plt.show()


# 添加用于检查空间对齐的辅助函数
def check_dsm_rgb_alignment(dataset_path='data/vaihingen/train1gemi', num_samples=3):
    """验证DSM与RGB空间对齐的辅助函数"""
    dataset = VaihingenDataset(data_root=dataset_path, use_dsm=True)
    
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        img = sample['img'].permute(1, 2, 0).numpy()  # CHW -> HWC
        dsm = sample['dsm'].squeeze().numpy()         # 移除通道维度
        mask = sample['gt_semantic_seg'].numpy()
        img_id = sample['img_id']
        
        # 显示图像、掩码和DSM，检查它们的对齐情况
        show_img_mask_dsm(img, mask, dsm, img_id)
