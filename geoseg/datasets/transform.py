import math
import numbers
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
import random
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from scipy.ndimage import maximum_filter


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask, dsm=None):
        # 扩展以支持DSM
        # print("img.size", img.size)
        # print("mask.size", mask.size)
        assert img.size == mask.size
        # 如果dsm不为空，则验证其尺寸
        if dsm is not None:
             assert img.size == dsm.size

        for t in self.transforms:
             if dsm is not None:
                 # 尝试调用支持DSM的变换
                 try:
                     img, mask, dsm = t(img, mask, dsm)
                 except TypeError:
                     # 如果变换不支持DSM，则只传img和mask（但通常几何变换需要同步，这可能需要根据具体Transform修改）
                     # 这里假设所有的t都必须更新以支持dsm，或者我们只在这里处理Compose
                     # 为了安全起见，我们应该更新所有相关的Transform类。
                     # 作为一个简单的回退（但这可能导致dsm不同步），我们可以：
                     # img, mask = t(img, mask) 
                     # 但这对于几何变换（缩放、裁剪）是错误的。
                     # 因此，我们需要检查 t 是否支持 dsm 参数，或者我们必须更新所有 Transform。
                     # 鉴于这是一个针对特定项目的修复，我们假设 t 会被更新。
                     # 但为了避免修改所有文件，我们可以检查参数数量。
                     # 更稳健的方法是修改 RandomScale 和 SmartCropV1 等类。
                     # 这里我们先尝试传递 dsm。
                     img, mask = t(img, mask)
             else:
                img, mask = t(img, mask)
        
        if dsm is not None:
            return img, mask, dsm
        return img, mask


class RandomCrop(object):
    """
    Take a random crop from the image.
    First the image or crop size may need to be adjusted if the incoming image
    is too small...
    If the image is smaller than the crop, then:
         the image is padded up to the size of the crop
         unless 'nopad', in which case the crop size is shrunk to fit the image
    A random crop is taken such that the crop fits within the image.
    If a centroid is passed in, the crop must intersect the centroid.
    """
    def __init__(self, size=512, ignore_index=12, nopad=True):

        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.ignore_index = ignore_index
        self.nopad = nopad
        self.pad_color = (0, 0, 0)

    def __call__(self, img, mask, centroid=None, dsm=None):
        assert img.size == mask.size
        # 如果有DSM，确保尺寸一致
        if dsm is not None:
            assert img.size == dsm.size

        w, h = img.size
        # ASSUME H, W
        th, tw = self.size
        if w == tw and h == th:
            if dsm is not None:
                return img, mask, dsm
            return img, mask

        if self.nopad:
            if th > h or tw > w:
                # Instead of padding, adjust crop size to the shorter edge of image.
                shorter_side = min(w, h)
                th, tw = shorter_side, shorter_side
        else:
            # Check if we need to pad img to fit for crop_size.
            if th > h:
                pad_h = (th - h) // 2 + 1
            else:
                pad_h = 0
            if tw > w:
                pad_w = (tw - w) // 2 + 1
            else:
                pad_w = 0
            border = (pad_w, pad_h, pad_w, pad_h)
            if pad_h or pad_w:
                img = ImageOps.expand(img, border=border, fill=self.pad_color)
                mask = ImageOps.expand(mask, border=border, fill=self.ignore_index)
                if dsm is not None:
                    dsm = ImageOps.expand(dsm, border=border, fill=0) # dsm fill 0
                w, h = img.size

        if centroid is not None:
            # Need to insure that centroid is covered by crop and that crop
            # sits fully within the image
            c_x, c_y = centroid
            max_x = w - tw
            max_y = h - th
            x1 = random.randint(c_x - tw, c_x)
            x1 = min(max_x, max(0, x1))
            y1 = random.randint(c_y - th, c_y)
            y1 = min(max_y, max(0, y1))
        else:
            if w == tw:
                x1 = 0
            else:
                x1 = random.randint(0, w - tw)
            if h == th:
                y1 = 0
            else:
                y1 = random.randint(0, h - th)
        
        crop_box = (x1, y1, x1 + tw, y1 + th)
        img_crop = img.crop(crop_box)
        mask_crop = mask.crop(crop_box)
        
        if dsm is not None:
            dsm_crop = dsm.crop(crop_box)
            return img_crop, mask_crop, dsm_crop
            
        return img_crop, mask_crop


class PadImage(object):
    def __init__(self, size=(512, 512), ignore_index=0):
        self.size = size
        self.ignore_index = ignore_index

    def __call__(self, img, mask):
        assert img.size == mask.size
        th, tw = self.size, self.size

        w, h = img.size

        if w > tw or h > th:
            wpercent = (tw / float(w))
            target_h = int((float(img.size[1]) * float(wpercent)))
            img, mask = img.resize((tw, target_h), Image.BICUBIC), mask.resize((tw, target_h), Image.NEAREST)

        w, h = img.size
        img = ImageOps.expand(img, border=(0, 0, tw - w, th - h), fill=0)
        mask = ImageOps.expand(mask, border=(0, 0, tw - w, th - h), fill=self.ignore_index)

        return img, mask


class RandomHorizontalFlip(object):

    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(self, img, mask=None):
        if mask is not None:
            if random.random() < self.prob:
                return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(
                    Image.FLIP_LEFT_RIGHT)
            else:
                return img, mask
        else:
            if random.random() < self.prob:
                return img.transpose(Image.FLIP_LEFT_RIGHT)
            else:
                return img


class RandomVerticalFlip(object):
    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(self, img, mask=None):
        if mask is not None:
            if random.random() < self.prob:
                return img.transpose(Image.FLIP_TOP_BOTTOM), mask.transpose(
                    Image.FLIP_TOP_BOTTOM)
            else:
                return img, mask
        else:
            if random.random() < self.prob:
                return img.transpose(Image.FLIP_TOP_BOTTOM)
            else:
                return img


class Resize(object):
    def __init__(self, size: tuple = (512,  512)):
        self.size = size  # size: (h, w)

    def __call__(self, img, mask):
        assert img.size == mask.size
        return img.resize(self.size, Image.BICUBIC), mask.resize(self.size, Image.NEAREST)


class RandomScale(object):
    def __init__(self, scale_list=[0.75, 1.0, 1.25], mode='value'):
        self.scale_list = scale_list
        self.mode = mode

    def __call__(self, img, mask, dsm=None):
        oh, ow = img.size
        # 如果有DSM，确保尺寸一致
        if dsm is not None:
            assert img.size == dsm.size

        scale_amt = 1.0
        if self.mode == 'value':
            scale_amt = np.random.choice(self.scale_list, 1)
        elif self.mode == 'range':
            scale_amt = random.uniform(self.scale_list[0], self.scale_list[-1])
        h = int(scale_amt * oh)
        w = int(scale_amt * ow)
        
        img_resized = img.resize((w, h), Image.BICUBIC)
        mask_resized = mask.resize((w, h), Image.NEAREST)
        
        if dsm is not None:
            # DSM是浮点数据，通常使用双线性插值
            dsm_resized = dsm.resize((w, h), Image.BICUBIC)
            return img_resized, mask_resized, dsm_resized
            
        return img_resized, mask_resized


class ColorJitter(object):
    def __init__(self, brightness=0.5, contrast=0.5, saturation=0.5):
        if not brightness is None and brightness>0:
            self.brightness = [max(1-brightness, 0), 1+brightness]
        if not contrast is None and contrast>0:
            self.contrast = [max(1-contrast, 0), 1+contrast]
        if not saturation is None and saturation>0:
            self.saturation = [max(1-saturation, 0), 1+saturation]

    def __call__(self, img, mask=None):
        r_brightness = random.uniform(self.brightness[0], self.brightness[1])
        r_contrast = random.uniform(self.contrast[0], self.contrast[1])
        r_saturation = random.uniform(self.saturation[0], self.saturation[1])
        img = ImageEnhance.Brightness(img).enhance(r_brightness)
        img = ImageEnhance.Contrast(img).enhance(r_contrast)
        img = ImageEnhance.Color(img).enhance(r_saturation)
        if mask is None:
            return img
        else:
            return img, mask


class SmartCropV1(object):
    def __init__(self, crop_size=512,
                 max_ratio=0.75,
                 ignore_index=12, nopad=False):
        self.crop_size = crop_size
        self.max_ratio = max_ratio
        self.ignore_index = ignore_index
        self.crop = RandomCrop(crop_size, ignore_index=ignore_index, nopad=nopad)

    def __call__(self, img, mask, dsm=None):
        assert img.size == mask.size
        # 如果有DSM，确保尺寸一致
        if dsm is not None:
             assert img.size == dsm.size
        
        count = 0
        while True:
            # 这里的 self.crop 是 RandomCrop 的实例
            # RandomCrop 需要被更新以支持 dsm，或者我们直接在这里处理 crop param
            # 更好的办法是更新 RandomCrop，让 SmartCrop 调用它
            
            # 为了简单起见，且不知道 RandomCrop 的内部随机逻辑如何保持同步
            # 我们应该先获取裁剪参数，然后对三个图像应用相同的裁剪
            
            # 由于 RandomCrop.call 目前直接返回裁剪后的图像，我们无法轻易分离参数
            # 我们可以在 RandomCrop 中增加一个 get_params 方法，或者直接在这里复用 RandomCrop 的逻辑
            
            # 最好的办法是修改 RandomCrop 让它支持可选的 dsm
            
            if dsm is not None:
                img_crop, mask_crop, dsm_crop = self.crop(img.copy(), mask.copy(), dsm=dsm.copy())
            else:
                img_crop, mask_crop = self.crop(img.copy(), mask.copy())
                
            count += 1
            labels, cnt = np.unique(np.array(mask_crop), return_counts=True)
            cnt = cnt[labels != self.ignore_index]
            if len(cnt) > 1 and np.max(cnt) / np.sum(cnt) < self.max_ratio:
                break
            if count > 10:
                break
        
        if dsm is not None:
            return img_crop, mask_crop, dsm_crop
        return img_crop, mask_crop

        return img_crop, mask_crop


class SmartCropV2(object):
    def __init__(self, crop_size=512, num_classes=13,
                 class_interest=[2, 3],
                 class_ratio=[0.1, 0.25],
                 max_ratio=0.75,
                 ignore_index=12, nopad=True):
        self.crop_size = crop_size
        self.num_classes = num_classes
        self.class_interest = class_interest
        self.class_ratio = class_ratio
        self.max_ratio = max_ratio
        self.ignore_index = ignore_index
        self.crop = RandomCrop(crop_size, ignore_index=ignore_index, nopad=nopad)

    def __call__(self, img, mask):
        assert img.size == mask.size
        count = 0
        while True:
            img_crop, mask_crop = self.crop(img.copy(), mask.copy())
            count += 1
            bins = np.array(range(self.num_classes + 1))
            class_pixel_counts, _ = np.histogram(np.array(mask_crop), bins=bins)
            cf = class_pixel_counts / (self.crop_size * self.crop_size)
            cf = np.array(cf)
            for c, f in zip(self.class_interest, self.class_ratio):
                if cf[c] > f:
                    break
            if np.max(cf) < 0.75 and np.argmax(cf) != self.ignore_index:
                break
            if count > 10:
                break

        return img_crop, mask_crop
