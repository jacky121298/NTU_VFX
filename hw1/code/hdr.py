import cv2
import glob
import random
import pathlib
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def sample_intensity(c_imgs: np.ndarray, n_sample: int):
    middle_img = c_imgs[c_imgs.shape[0] // 2, ...]
    c_sample = np.zeros((n_sample, c_imgs.shape[0]), dtype=np.uint8)
    for i in range(256):
        rows, cols = np.where(middle_img == i)
        if len(rows) > 0:
            idx = random.randrange(len(rows))
            r, c = rows[idx], cols[idx]
            for j in range(c_imgs.shape[0]):
                c_sample[i, j] = c_imgs[j, r, c]
    return c_sample

def getResponseCurve(c_sample: np.ndarray, delta_t: list, weights: list, smooth_lambda: np.float64):
    n_sample, n_image = c_sample.shape
    A = np.zeros((n_sample * n_image + 1 + 254, 256 + n_sample), dtype=np.float64)
    b = np.zeros((A.shape[0], 1), dtype=np.float64)

    row_idx = 0
    for i in range(n_sample):
        for j in range(n_image):
            w = weights[c_sample[i, j]]
            A[row_idx, c_sample[i, j]] = w
            A[row_idx, 256 + i] = -w
            b[row_idx] = w * delta_t[j]
            row_idx += 1

    A[row_idx, 128] = 1
    row_idx += 1

    for i in range(1, 255):
        w = weights[i]
        A[row_idx, i - 1] = smooth_lambda * w
        A[row_idx,     i] = smooth_lambda * w * -2
        A[row_idx, i + 1] = smooth_lambda * w
        row_idx += 1

    pinv_A = np.linalg.pinv(A)
    x = np.dot(pinv_A, b) # (256 + n_sample, 1)
    return x[:256].squeeze()

def getRadianceMap(c_imgs: list, c_g: np.ndarray, delta_t: list, weights: list):
    n_image = len(c_imgs)
    img_shape = c_imgs[0].shape
    c_E = np.zeros(img_shape, dtype=np.float64)
    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            g = np.array([c_g[c_imgs[k][i, j]] for k in range(n_image)])
            w = np.array([weights[c_imgs[k][i, j]] for k in range(n_image)])
            if np.sum(w) == 0:
                c_E[i, j] = g[n_image // 2] - delta_t[n_image // 2]
            else:
                c_E[i, j] = np.sum(w * (g - np.array(delta_t)) / np.sum(w))
    return c_E

def getHDR(imgs: list, delta_t: list, weights: list):
    n_image = len(imgs)
    n_sample = 256
    
    response_curve = []
    radiance_map = np.zeros(imgs[0].shape, dtype=np.float64)
    for c in range(imgs[0].shape[2]):
        print(f'======== Channel {c} ========')
        c_imgs = np.stack([img[:, :, c] for img in imgs]) # (n_image, h, w)
        print(f'> Sample intensity')
        c_sample = sample_intensity(c_imgs, n_sample) # (n_sample, n_image)
        print(f'> Compute response curve')
        c_g = getResponseCurve(c_sample, delta_t, weights, smooth_lambda=100.0)
        response_curve.append(c_g)
        print(f'> Compute radiance map')
        c_E = getRadianceMap(c_imgs, c_g, delta_t, weights)
        radiance_map[:, :, c] = c_E

        print(f'============ Done ===========')
    return response_curve, radiance_map

if __name__ == '__main__':
    # python3 hdr.py --input_dir ../mtb
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', help='Path to the input image dir.')
    args = parser.parse_args()

    mtb_img, delta_t = [], []
    for img_path in sorted(glob.glob(args.input_dir + '/*.jpg')):
        mtb_img.append(cv2.imread(img_path).astype(np.uint8))
        delta_t.append(np.log(float(img_path.split('_')[-1].rstrip('.jpg')) / 10))
    
    z_min, z_max = 0, 255
    weights = [z - z_min for z in range((z_min + z_max) // 2 + 1)] + [z_max - z for z in range((z_min + z_max) // 2 + 1, z_max + 1)]
    response_curve, radiance_map = getHDR(mtb_img, delta_t, weights)
    np.save('../images/hdr.npy', radiance_map)
    np.save('../images/crc.npy', np.array(response_curve))