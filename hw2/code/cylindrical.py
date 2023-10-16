import cv2
import glob
import random
import pathlib
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def bilinear_interpolate(img: np.ndarray, r: np.ndarray, c: np.ndarray):
    r0 = np.floor(r).astype(np.int)
    c0 = np.floor(c).astype(np.int)
    r1 = r0 + 1
    c1 = c0 + 1

    r0 = np.clip(r0, 0, img.shape[0] - 1)
    r1 = np.clip(r1, 0, img.shape[0] - 1)
    c0 = np.clip(c0, 0, img.shape[1] - 1)
    c1 = np.clip(c1, 0, img.shape[1] - 1)

    Ia = img[r0, c0]
    Ib = img[r0, c1]
    Ic = img[r1, c1]
    Id = img[r1, c0]

    wa = np.expand_dims((r1 - r) * (c1 - c), axis=1)
    wb = np.expand_dims((r1 - r) * (c - c0), axis=1)
    wc = np.expand_dims((r - r0) * (c - c0), axis=1)
    wd = np.expand_dims((r - r0) * (c1 - c), axis=1)
    
    warped_img = (wa * Ia) + (wb * Ib) + (wc * Ic) + (wd * Id)
    return warped_img.reshape(img.shape)

def cylindrical(bgr_img, s, savefig=False):
    n = len(bgr_img)
    h, w, c = bgr_img[0].shape
    cyl_img = np.zeros((n, h, w, c)).astype(np.float32)

    cyl_x, cyl_y = np.meshgrid(np.arange(w), np.arange(h))
    cyl_x = cyl_x.flatten() - w // 2
    cyl_y = cyl_y.flatten() - h // 2

    for i in tqdm(range(n)):
        x = s * np.tan(cyl_x / s)
        y = cyl_y / s * np.sqrt(x ** 2 + s ** 2)
        x += w // 2
        y += h // 2

        cyl_img[i] = bilinear_interpolate(bgr_img[i], y, x)
        if savefig:
            cv2.imwrite(f'{args.output_dir}/cyl_{i:02d}.jpg', cyl_img[i])
    
    return cyl_img.astype(np.uint8)

if __name__ == '__main__':
    # python3 cylindrical.py --input_dir ../lake_data --h 6720 --focal 7000.0 --output_dir ../lake
    # python3 cylindrical.py --input_dir ../mountain_data --h 5312 --focal 4800.0 --output_dir ../mountain
    parser = argparse.ArgumentParser(description='Cylindrical projection.')
    parser.add_argument('--input_dir', help='Path to the input image dir.')
    parser.add_argument('--h', type=int, default=6720, help='Resize input image height.')
    parser.add_argument('--focal', type=float, default=7000.0, help='Focal length of camera.')
    parser.add_argument('--output_dir', help='Path to the output image dir.')
    args = parser.parse_args()

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bgr_img = []
    for img_path in sorted(glob.glob(args.input_dir + '/*.jpg')):
        if 'photoshop' in img_path:
            continue

        bgr = cv2.imread(img_path)
        if bgr.shape[0] > args.h:
            w = int(bgr.shape[1] * args.h / bgr.shape[0])
            bgr = cv2.resize(bgr, (w, args.h), cv2.INTER_LINEAR)
        bgr_img.append(bgr)

    print('Start cylindrical projection')
    cylindrical(bgr_img, s=args.focal, savefig=True)