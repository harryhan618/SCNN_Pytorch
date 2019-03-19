import numpy as np
import cv2


def getLane(prob_map, pts, thresh, resize_shape=None):
    """
    Arguments:
    ----------
    prob_map: prob map for single lane, np array size (h, w)
    resize_shape:  reshape size target, (H, W)

    Return:
    ----------
    coords: x coords bottom up every 20px, 0 for non-exist, in resized shape
    """
    if resize_shape is None:
        resize_shape = prob_map.shape
    h, w = prob_map.shape
    H, W = resize_shape

    coords = np.zeros(pts)
    for i in range(pts):
        lindId = int(h - (i-1)*20 / H * h)
        line = prob_map[lindId, :]
        id = np.argmax(line)
        if line[id] > thresh:
            coords[i] = id
    if (coords>0).sum() < 2:
        coords = np.zeros(18)
    return coords


def prob2lines(seg_pred, exist, resize_shape=None, smooth=True, pts=18, thresh=0.3):
    """
    Arguments:
    ----------
    seg_pred: np.array size (5, H, W)
    exist:   list of existence, e.g. [0, 1, 1, 0]
    smooth:  whether to smooth the probability or not
    pts:     how many points for one lane
    thresh:  probability threshold

    Return:
    ----------
    coordinates: np.array size (4, pts)
    """
    coordinates = np.zeros((4, pts))

    seg_pred = np.ascontiguousarray(np.transpose(seg_pred, (1, 2, 0)))
    for i in range(4):
        prob_map = seg_pred[..., i]
        if smooth:
            prob_map = cv2.GaussianBlur(prob_map, (9, 9), 0)
        if exist[i]>0:
            coordinates[i] = getLane(prob_map, pts, thresh, resize_shape)
    return coordinates

