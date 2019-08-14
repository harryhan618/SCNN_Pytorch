import random
import numpy as np
import cv2

from utils.transforms.transforms import CustomTransform


class RandomFlip(CustomTransform):
    def __init__(self, prob_x=0, prob_y=0):
        """
        Arguments:
        ----------
        prob_x: range [0, 1], probability to use horizontal flip, setting to 0 means disabling flip
        prob_y: range [0, 1], probability to use vertical flip
        """
        self.prob_x = prob_x
        self.prob_y = prob_y

    def __call__(self, sample):
        img = sample.get('img').copy()
        segLabel = sample.get('segLabel', None)
        if segLabel is not None:
            segLabel = segLabel.copy()

        flip_x = np.random.choice([False, True], p=(1 - self.prob_x, self.prob_x))
        flip_y = np.random.choice([False, True], p=(1 - self.prob_y, self.prob_y))
        if flip_x:
            img = np.ascontiguousarray(np.flip(img, axis=1))
            if segLabel is not None:
                segLabel = np.ascontiguousarray(np.flip(segLabel, axis=1))

        if flip_y:
            img = np.ascontiguousarray(np.flip(img, axis=0))
            if segLabel is not None:
                segLabel = np.ascontiguousarray(np.flip(segLabel, axis=0))

        _sample = sample.copy()
        _sample['img'] = img
        _sample['segLabel'] = segLabel
        return _sample


class Darkness(CustomTransform):
    def __init__(self, coeff):
        assert coeff >= 1., "Darkness coefficient must be greater than 1"
        self.coeff = coeff

    def __call__(self, sample):
        img = sample.get('img')
        coeff = np.random.uniform(1., self.coeff)
        img = (img.astype('float32') / coeff).astype('uint8')

        _sample = sample.copy()
        _sample['img'] = img
        return _sample