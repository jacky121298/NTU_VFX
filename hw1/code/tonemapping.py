import cv2
import glob
import pathlib
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def gammaToneMapping(img: np.ndarray, gamma: np.float):
    gamma_img = np.power(img/255.0, 1.0/gamma) * 255.0
    return gamma_img

def luminosity(bgr_img: np.ndarray):
    bgr_weights = np.array([1 / 61, 40 / 61, 20 / 61])
    lum_img = bgr_weights[0] * bgr_img[..., 0] + bgr_weights[1] * bgr_img[..., 1] + bgr_weights[2] * bgr_img[..., 2]
    return lum_img

def rec_filter_horizontal(I: np.ndarray, D: np.ndarray, sigma: np.float):
    a = np.exp(-np.sqrt(2.0) / sigma)
    V = np.power(a, D)
    
    F = I.copy()
    h, w, num_channels = I.shape
    for i in range(1, w):
        for c in range(num_channels):
            F[:,i,c] = F[:, i, c] + V[:, i] * (F[:, i-1, c] - F[:, i, c])
    
    for i in range(w - 2, -1, -1):
        for c in range(num_channels):
            F[:,i,c] = F[:, i, c] + V[:, i+1] * (F[:, i+1, c] - F[:, i, c])
    return F

def fast_bilateral(I: np.ndarray, sigma_s: np.float, sigma_r: np.float, num_iterations: np.int=5, J: np.ndarray=None):
    if I.ndim == 3:
        img = I.copy()
    else:
        h, w = I.shape
        img = I.reshape((h, w, 1))
    
    if J is None:
        J = img
    if J.ndim == 2:
        h, w = J.shape
        J = np.reshape(J, (h, w, 1))

    h, w, num_channels = J.shape
    dIcdx = np.diff(J, n=1, axis=1)
    dIcdy = np.diff(J, n=1, axis=0)
    dIdx = np.zeros((h, w))
    dIdy = np.zeros((h, w))

    for c in range(num_channels):
        dIdx[:, 1:] = dIdx[:, 1:] + np.abs(dIcdx[:, :, c])
        dIdy[1:, :] = dIdy[1:, :] + np.abs(dIcdy[:, :, c])
    
    dHdx = (1.0 + sigma_s / sigma_r * dIdx)
    dVdy = (1.0 + sigma_s / sigma_r * dIdy)
    dVdy = dVdy.T
    
    F = img.copy()
    N = num_iterations
    for i in range(num_iterations):
        sigma_H_i = sigma_s * np.sqrt(3.0) * (2.0 ** (N - (i + 1))) / np.sqrt(4.0 ** N - 1.0)
        F = rec_filter_horizontal(F, dHdx, sigma_H_i)
        F = np.swapaxes(F, 0, 1)
        F = rec_filter_horizontal(F, dVdy, sigma_H_i)
        F = np.swapaxes(F, 0, 1)
    return F

def bilateral_filter(pad_img: np.ndarray, i: np.int, j: np.int, radius: np.int, sigma_s: np.float, sigma_r: np.float):
    total, weights = 0, 0
    pad_width = radius // 2
    for k in range(i - pad_width, i + pad_width + 1):
        for l in range(j - pad_width, j + pad_width + 1):
            spatial_kernel = (np.power(i - k, 2) + np.power(j - l, 2)) / (2 * np.power(sigma_s, 2))
            range_kernel = np.absolute(pad_img[i, j] - pad_img[k, l]) / (2 * np.power(sigma_r, 2))
            weight = np.exp(-spatial_kernel - range_kernel)
            total += (weight * pad_img[k, l])
            weights += weight
    return total / weights

def bilateral(img: np.ndarray, sigma_s: np.float, sigma_r: np.float, radius: np.int=7):
    pad_width = radius // 2
    ret_img = np.zeros(img.shape)
    pad_img = np.pad(np.copy(img), pad_width=pad_width, mode='mean')
    for i in tqdm(range(pad_width, pad_img.shape[0] - pad_width + 1)):
        for j in range(pad_width, pad_img.shape[1] - pad_width + 1):
            ret_img[i - pad_width, j - pad_width] = bilateral_filter(pad_img, i, j, radius, sigma_s, sigma_r)
    return ret_img

def luminosity_separation(lum_img: np.ndarray, sigma_s: np.float=0.02, sigma_r: np.float=0.4, eps: np.float=1e-10):
    sigma_s = np.max(lum_img.shape) * sigma_s
    lum_base = fast_bilateral(np.log10(lum_img + eps), sigma_s, sigma_r)
    lum_base = np.squeeze(np.power(10.0, lum_base) - eps)
    lum_detail = lum_img / lum_base
    return lum_base, lum_detail

def durand(img: np.ndarray, ratio: np.float=0.4):
    lum_img = luminosity(img)
    cv2.imwrite('../images/lum.jpg', lum_img * 255.0)
    clr_img = img / lum_img[..., np.newaxis]
    cv2.imwrite('../images/clr.jpg', clr_img * 255.0)

    lum_base, lum_detail = luminosity_separation(lum_img)
    cv2.imwrite('../images/lum_base.jpg', lum_base * 255.0)
    cv2.imwrite('../images/lum_detail.jpg', lum_detail * 255.0)   
    
    lum2_img = np.power(10.0, ratio * np.log10(lum_base) + np.log10(lum_detail))
    cv2.imwrite('../images/lum2.jpg', lum2_img * 255.0)
    durand_img = clr_img * lum2_img[..., np.newaxis]
    durand_img = np.clip(durand_img, 0, 1) * 255
    return durand_img

if __name__ == '__main__':
    # python3 tonemapping.py --input_dir ../mtb --hdr ../images/hdr.npy
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', help='Path to the input image dir.')
    parser.add_argument('--hdr', help='Path to the hdr image.')
    args = parser.parse_args()

    mtb_img = []
    for img_path in sorted(glob.glob(args.input_dir + '/*.jpg')):
        mtb_img.append(cv2.imread(img_path).astype(np.uint8))
    hdr = np.exp(np.load(args.hdr)).astype(np.float32) # (h, w, c)

    # Plot HDR image
    plt.figure(figsize=(8, 4))
    plt.imshow(cv2.cvtColor(hdr, cv2.COLOR_BGR2GRAY), cmap='jet')
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('../images/hdr.jpg')
    plt.close()

    # Plot gamma tone mapping
    gamma_img = gammaToneMapping(hdr, gamma=5.0)
    cv2.imwrite('../images/gamma.jpg', gamma_img)

    # Plot durand tone mapping
    durand_img = durand(hdr, ratio=0.4)
    cv2.imwrite('../images/durand.jpg', durand_img)