# code in this file is adpated from
# https://github.com/ildoonet/pytorch-randaugment/blob/master/RandAugment/augmentations.py
# https://github.com/google-research/fixmatch/blob/master/third_party/auto_augment/augmentations.py
# https://github.com/google-research/fixmatch/blob/master/libml/ctaugment.py
import logging
import random

import numpy as np
import PIL
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.ImageDraw
from PIL import Image

logger = logging.getLogger(__name__)

PARAMETER_MAX = 10
RESAMPLE_MODE = Image.BICUBIC
FILL_COLOR = (128, 128, 128)


def AutoContrast(img, **kwarg):
    return PIL.ImageOps.autocontrast(img)


def Brightness(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Color(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Color(img).enhance(v)


def Contrast(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def Cutout(img, v, max_v, **kwarg):
    if v == 0:
        return img
    v = _float_parameter(v, max_v)
    v = int(v * min(img.size))

    w, h = img.size
    x0 = np.random.uniform(0, w)
    y0 = np.random.uniform(0, h)
    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = int(min(w, x0 + v))
    y1 = int(min(h, y0 + v))
    xy = (x0, y0, x1, y1)
    # gray
    color = FILL_COLOR
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img


def CutoutConst(img, v, max_v, **kwarg):
    v = _int_parameter(v, max_v)
    w, h = img.size
    x0 = np.random.uniform(0, w)
    y0 = np.random.uniform(0, h)
    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = int(min(w, x0 + v))
    y1 = int(min(h, y0 + v))
    xy = (x0, y0, x1, y1)
    # gray
    color = FILL_COLOR
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img


def Equalize(img, **kwarg):
    return PIL.ImageOps.equalize(img)


def Identity(img, **kwarg):
    return img


def Invert(img, **kwarg):
    return PIL.ImageOps.invert(img)


def Posterize(img, v, max_v, bias=0):
    v = 8 - _round_parameter(v, max_v) + bias
    return PIL.ImageOps.posterize(img, v)


def Rotate(img, v, max_v, **kwarg):
    v = _float_parameter(v, max_v)
    if random.random() < 0.5:
        v = -v
    return img.rotate(v, RESAMPLE_MODE, fillcolor=FILL_COLOR)


def Sharpness(img, v, max_v, bias):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def ShearX(img, v, max_v, **kwarg):
    v = _float_parameter(v, max_v)
    if random.random() < 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0), RESAMPLE_MODE, fillcolor=FILL_COLOR)


def ShearY(img, v, max_v, **kwarg):
    v = _float_parameter(v, max_v)
    if random.random() < 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0), RESAMPLE_MODE, fillcolor=FILL_COLOR)


def Solarize(img, v, max_v, **kwarg):
    v = _int_parameter(v, max_v)
    return PIL.ImageOps.solarize(img, 255 - v)


def SolarizeAdd(img, v, max_v, threshold=128, **kwarg):
    v = _int_parameter(v, max_v)
    img_np = np.array(img).astype(np.int)
    img_np = img_np + v
    img_np = np.clip(img_np, 0, 255)
    img_np = img_np.astype(np.uint8)
    img = Image.fromarray(img_np)
    return PIL.ImageOps.solarize(img, threshold)


def TranslateX(img, v, max_v, **kwarg):
    v = _float_parameter(v, max_v)
    if random.random() < 0.5:
        v = -v
    v = int(v * img.size[0])
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0), RESAMPLE_MODE, fillcolor=FILL_COLOR)


def TranslateY(img, v, max_v, **kwarg):
    v = _float_parameter(v, max_v)
    if random.random() < 0.5:
        v = -v
    v = int(v * img.size[1])
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v), RESAMPLE_MODE, fillcolor=FILL_COLOR)


def TranslateXConst(img, v, max_v, **kwarg):
    v = _float_parameter(v, max_v)
    if random.random() < 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0), RESAMPLE_MODE, fillcolor=FILL_COLOR)


def TranslateYConst(img, v, max_v, **kwarg):
    v = _float_parameter(v, max_v)
    if random.random() < 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v), RESAMPLE_MODE, fillcolor=FILL_COLOR)


def _float_parameter(v, max_v):
    return float(v) * max_v / PARAMETER_MAX


def _int_parameter(v, max_v):
    return int(v * max_v / PARAMETER_MAX)


def _round_parameter(v, max_v):
    return int(round(v * max_v / PARAMETER_MAX))


def rand_augment_pool():
    # Test
    augs = [
        (AutoContrast, None, None),
        (Equalize, None, None),
        (Invert, None, None),
        (Rotate, 30, None),
        (Posterize, 4, 0),
        (Solarize, 256, None),
        (Color, 1.8, 0.1),
        (Contrast, 1.8, 0.1),
        (Brightness, 1.8, 0.1),
        (Sharpness, 1.8, 0.1),
        (ShearX, 0.3, None),
        (ShearY, 0.3, None),
        (TranslateXConst, 100, None),
        (TranslateYConst, 100, None),
        (CutoutConst, 40, None),  # Use RandomErasing instead of Cutout.
    ]
    return augs


def fixmatch_augment_pool():
    # FixMatch paper
    augs = [
        (AutoContrast, None, None),
        (Brightness, 0.9, 0.05),
        (Color, 0.9, 0.05),
        (Contrast, 0.9, 0.05),
        (Equalize, None, None),
        (Identity, None, None),
        (Posterize, 4, 4),
        (Rotate, 30, 0),
        (Sharpness, 0.9, 0.05),
        (ShearX, 0.3, 0),
        (ShearY, 0.3, 0),
        (Solarize, 256, None),
        (TranslateX, 0.3, 0),
        (TranslateY, 0.3, 0)
    ]
    return augs


def cifar_augment_pool():
    # Test
    augs = [
        (AutoContrast, None, None),
        (Equalize, None, None),
        (Invert, None, None),
        (Rotate, 30, None),
        (Posterize, 4, 0),
        (Solarize, 256, None),
        (Color, 1.8, 0.1),
        (Contrast, 1.8, 0.1),
        (Brightness, 1.8, 0.1),
        (Sharpness, 1.8, 0.1),
        (ShearX, 0.3, None),
        (ShearY, 0.3, None),
        (TranslateXConst, 32 // 8, None),
        (TranslateYConst, 32 // 8, None),
        (CutoutConst, 32 // 8, None),
    ]
    return augs


def soft_augment_pool():
    # Test
    augs = [(AutoContrast, None, None),
            (Brightness, 1.8, 0.1),
            (Color, 1.8, 0.1),
            (Contrast, 1.8, 0.1),
            (CutoutConst, 40, None),
            (Posterize, 4, 0),
            (Sharpness, 1.8, 0.1),
            ]
    return augs


class SoftAugment(object):
    def __init__(self, n, m, resample_mode=PIL.Image.BICUBIC):
        global RESAMPLE_MODE
        RESAMPLE_MODE = resample_mode
        self.n = n
        self.m = m
        self.augment_pool = soft_augment_pool()

    def __call__(self, img):
        ops = random.choices(self.augment_pool, k=self.n)
        for op, max_v, bias in ops:
            prob = np.random.uniform(0.2, 0.8)
            if random.random() + prob >= 1:
                img = op(img, v=self.m, max_v=max_v, bias=bias)
        return img


class RandAugment(object):
    def __init__(self, n, m, resample_mode=PIL.Image.BICUBIC, fill=(128, 128, 128)):
        global RESAMPLE_MODE, FILL_COLOR
        RESAMPLE_MODE = resample_mode
        FILL_COLOR = fill
        self.n = int(n)
        self.m = m
        self.mstd = mstd
        self.augment_pool = rand_augment_pool()

    def __call__(self, img):
        ops = random.choices(self.augment_pool, k=self.n)
        for op, max_v, bias in ops:
            prob = np.random.uniform(0.2, 0.8)
            if random.random() <= prob:
                img = op(img, v=self.m, max_v=max_v, bias=bias)
        return img


class RandAugmentCIFAR(object):
    def __init__(self, n, m, resample_mode=PIL.Image.BICUBIC, fill=(128, 128, 128)):
        global RESAMPLE_MODE, FILL_COLOR
        RESAMPLE_MODE = resample_mode
        FILL_COLOR = fill
        self.n = int(n)
        self.m = m
        self.augment_pool = cifar_augment_pool()

    def __call__(self, img):
        ops = random.choices(self.augment_pool, k=self.n)
        for op, max_v, bias in ops:
            prob = np.random.uniform(0.2, 0.8)
            if random.random() <= prob:
                img = op(img, v=self.m, max_v=max_v, bias=bias)
        img = CutoutConst(img, v=self.m, max_v=32 // 4)
        return img


class RandAugmentMC(object):
    def __init__(self, n, m, resample_mode=PIL.Image.BICUBIC, fill=(128, 128, 128)):
        global RESAMPLE_MODE, FILL_COLOR
        RESAMPLE_MODE = resample_mode
        FILL_COLOR = fill
        self.n = int(n)
        self.m = m
        self.augment_pool = fixmatch_augment_pool()

    def __call__(self, img):
        ops = random.choices(self.augment_pool, k=self.n)
        for op, max_v, bias in ops:
            v = np.random.randint(1, self.m)
            if random.random() < 0.5:
                img = op(img, v=v, max_v=max_v, bias=bias)
        img = CutoutConst(img, 40)
        return img
