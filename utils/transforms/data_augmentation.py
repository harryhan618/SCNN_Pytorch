import random
import numpy as np
import cv2

from utils.transforms.transforms import CustomTransform


class SquarePad(CustomTransform):
    def __init__(self, mode='constant', value=0):
        mode_dict = {
            'constant': cv2.BORDER_CONSTANT,
            'replicate': cv2.BORDER_REPLICATE,
            'reflect': cv2.BORDER_REFLECT
        }
        self.mode = mode_dict[mode]
        self.value = value

    def __call__(self, img, boxes):
        H, W = img.shape[:2]
        diff = abs(H-W)
        if W >= H:
            img = cv2.copyMakeBorder(img, 0, diff, 0, 0, self.mode, value=self.value)
        else:
            img = cv2.copyMakeBorder(img, 0, 0, 0, diff, self.mode, value=self.value)
        boxes = boxes.copy()
        return img, boxes


class RandomFlip(CustomTransform):
    def __init__(self, px=0, py=0):
        """
        Arguments:
        ----------
        px: range [0, 1], probability to use horizontal flip
        py: range [0, 1], probability to use vertical flip
        """
        self.px = px
        self.py = py

    def __call__(self, img, boxes):
        """
        Arguments:
        ----------
        img: numpy array, shape (H, W, 3)
        boxes: numpy array: shape (N,4)
        """
        img = img.copy()
        boxes = boxes.copy()
        H, W = img.shape[:2]

        flip_x = np.random.choice([False, True], p=(1-self.px, self.px))
        flip_y = np.random.choice([False, True], p=(1-self.py, self.py))
        if flip_x:
            img = np.ascontiguousarray(np.flip(img, axis=1))
            boxes[:, ::2] = W - boxes[:, ::2]
            boxes[:, ::2] = boxes[:, -2::-2]

        if flip_y:
            img = np.ascontiguousarray(np.flip(img, axis=0))
            boxes[:, 1::2] = H - boxes[:, 1::2]
            boxes[:, 1::2] = boxes[:, -1::-2]
        return img, boxes


class RandomTranslate(CustomTransform):
    """
    Randomly translate img
    """
    def __init__(self, exceed_x=0.2, exceed_y=0.2, prob=0.5):
        """
        Argmuments:
        ----------
        exceed_x: percentage allowed for all boxes to exceed image board in x axis
        exceed_y: percentage allowed for all boxes to exceed image board in y axis
        prob: range [0, 1], larger value means larger probability to perform the operation
        """
        self.exceed_x = exceed_x
        self.exceed_y = exceed_y
        self.prob = prob

    def __call__(self, img, boxes):
        boxes = boxes.copy()

        rand_prob = random.random()
        if rand_prob <= self.prob:
            H, W = img.shape[:2]
            _img = np.zeros_like(img)

            offset_x_low = -np.min(boxes[:, 0]*(1-self.exceed_x) + boxes[:, 2]*self.exceed_x)
            offset_y_low = -np.min(boxes[:, 1] * (1 - self.exceed_y) + boxes[:, 3] * self.exceed_y)
            offset_x_high = W - np.max(boxes[:, 2] * (1 - self.exceed_x) + boxes[:, 0] * self.exceed_x)
            offset_y_high = H - np.max(boxes[:, 3] * (1 - self.exceed_y) + boxes[:, 1] * self.exceed_y)

            offset_x = random.randint(int(offset_x_low), int(offset_x_high))
            offset_y = random.randint(int(offset_y_low), int(offset_y_high))

            if offset_y<=0 and offset_x<=0:
                _img[:H+offset_y, :W+offset_x] = img[-offset_y:, -offset_x:]
            elif offset_y<=0 and offset_x>0:
                _img[:H+offset_y, offset_x:] = img[-offset_y:, :W-offset_x]
            elif offset_y>0 and  offset_x<=0:
                _img[offset_y:, :W+offset_x] = img[:H-offset_y, -offset_x:]
            else:
                _img[offset_y:, offset_x:] = img[:H-offset_y, :W-offset_x]

            boxes[:, ::2] += offset_x
            boxes[:, 1::2] += offset_y
            boxes[:, [0, 1]] = np.maximum(boxes[:, [0, 1]], 0.)
            boxes[:, 2] = np.minimum(boxes[:, 2], W)
            boxes[:, 3] = np.minimum(boxes[:, 3], H)
        else:
            _img = img.copy()
        return _img, boxes


class Random_Color_Distort(CustomTransform):
    def __init__(self, brightness=32, contrast=0.4, prob=0.5):
        self.brightness = brightness
        self.contrast = max(contrast, 1.)
        self.prob = prob

    def __call__(self, img, boxes=None):
        img = img.copy()

        rand_prob = random.random()
        if rand_prob <= self.prob:
            img = self._brightness(img)
            img = self._contrast(img)

        if boxes is not None:
            return img, boxes
        return img

    def _brightness(self, img):
        low = np.min(img)
        high = 255 - np.max(img)
        value = random.randint(-self.brightness, self.brightness)
        value = np.uint8(min(min(low, value), high))
        img += value
        return img

    def _contrast(self, img):
        value = random.uniform(1-self.contrast, 1+self.contrast)
        _img = img.copy().astype('float')
        _img *= max(value, np.max(_img / 255.))
        return _img.astype('uint8')
