import numpy as np
import cv2


def getLane(prob_map, y_px_gap, pts, thresh, resize_shape=None):
    """
    Arguments:
    ----------
    prob_map: prob map for single lane, np array size (h, w)
    resize_shape:  reshape size target, (H, W)

    Return:
    ----------
    coords: x coords bottom up every y_px_gap px, 0 for non-exist, in resized shape
    """
    if resize_shape is None:
        resize_shape = prob_map.shape
    h, w = prob_map.shape
    H, W = resize_shape

    coords = np.zeros(pts)
    for i in range(pts):
        y = int(h - i*y_px_gap / H * h - 1)
        if y < 0:
            break
        line = prob_map[y, :]
        id = np.argmax(line)
        if line[id] > thresh:
            coords[i] = int(id / w * W)
    if (coords>0).sum() < 2:
        coords = np.zeros(18)
    return coords


def prob2lines(seg_pred, exist, resize_shape=None, smooth=True, y_px_gap=20, pts=None, thresh=0.3):
    """
    Arguments:
    ----------
    seg_pred: np.array size (5, h, w)
    resize_shape:  reshape size target, (H, W)
    exist:   list of existence, e.g. [0, 1, 1, 0]
    smooth:  whether to smooth the probability or not
    y_px_gap: y pixel gap for sampling
    pts:     how many points for one lane
    thresh:  probability threshold

    Return:
    ----------
    coordinates: [x, y] list of lanes, e.g.: [ [[9, 569], [50, 549]] ,[[630, 569], [647, 549]] ]
    """
    if resize_shape is None:
        resize_shape = seg_pred.shape[1:] # seg_pred (5, h, w)
    _, h, w = seg_pred.shape
    H, W = resize_shape
    coordinates = []

    if pts is None:
        pts = round(H/2 / y_px_gap)

    seg_pred = np.ascontiguousarray(np.transpose(seg_pred, (1, 2, 0)))
    for i in range(4):
        prob_map = seg_pred[..., i+1]
        if smooth:
            prob_map = cv2.blur(prob_map, (9, 9), borderType=cv2.BORDER_REPLICATE)
        if exist[i]>0:
            coords = getLane(prob_map, y_px_gap, pts, thresh, resize_shape)
            coordinates.append([[coords[i], H-1-i*y_px_gap] for i in range(pts) if coords[i]>0])

    return coordinates

