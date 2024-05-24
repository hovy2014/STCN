import numpy as np
from PIL import Image
import cv2
import thinplate as tps
import torch

cv2.setNumThreads(0)

def pick_random_points(h, w, n_samples):
    y_idx = np.random.choice(np.arange(h), size=n_samples, replace=False)
    x_idx = np.random.choice(np.arange(w), size=n_samples, replace=False)
    return y_idx/h, x_idx/w


def warp_dual_cv(img, mask, c_src, c_dst):
    dshape = img.shape
    theta = tps.tps_theta_from_points(c_src, c_dst, reduced=True)
    grid = tps.tps_grid(theta, c_dst, dshape)
    mapx, mapy = tps.tps_grid_to_remap(grid, img.shape)
    return cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR), cv2.remap(mask, mapx, mapy, cv2.INTER_NEAREST)

def warp_dual_cv_torch(img, mask, c_src, c_dst):
    dshape = img.shape
    h,w,c = dshape
    theta = tps.tps_theta_from_points(c_src, c_dst, reduced=True)
    thetatorch = torch.from_numpy(theta[None,]).double()
    c_dsttorch = torch.from_numpy(c_dst[None,]).double()
    gridtorch = tps.torch.tps_grid(thetatorch, c_dsttorch, [1, c, h, w])
    grid = gridtorch.numpy()[0]
    grid = (grid + 1) / 2
    mapx, mapy = tps.tps_grid_to_remap(grid, img.shape)
    return cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR), cv2.remap(mask, mapx, mapy, cv2.INTER_NEAREST)

def random_tps_warp(img, mask, scale, n_ctrl_pts=12, tensor_flag=False):
    """
    Apply a random TPS warp of the input image and mask
    Uses randomness from numpy
    """
    img = np.asarray(img)
    mask = np.asarray(mask)

    h, w = mask.shape
    points = pick_random_points(h, w, n_ctrl_pts)
    c_src = np.stack(points, 1)
    c_dst = c_src + np.random.normal(scale=scale, size=c_src.shape)
    if tensor_flag:
        warp_im, warp_gt = warp_dual_cv_torch(img, mask, c_src, c_dst)
    else:
        warp_im, warp_gt = warp_dual_cv(img, mask, c_src, c_dst)

    return Image.fromarray(warp_im), Image.fromarray(warp_gt)

