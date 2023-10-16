import cv2
import glob
import random
import pathlib
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial.distance import cdist

def find_matches(desc1, desc2, thres=0.7):
    distance = cdist(desc1['d'], desc2['d'], metric='cosine')
    nn = np.argsort(distance, axis=1)
    
    matches = []
    for i, n in enumerate(nn):
        f = distance[i, n[0]]
        s = distance[i, n[1]]
        if f < s * thres:
            matches.append([i, n[0]])
    
    return matches

def ransac(matches, kp1, kp2, n=1, r=5, ransac_iter=1000):
    matches = np.array(matches)
    points1 = kp1[matches[:, 0]]
    points2 = kp2[matches[:, 1]]

    best_xy = None
    best_inlier = 0
    for i in range(ransac_iter):
        samples = np.random.randint(0, points1.shape[0], n)
        xy = np.mean(points1[samples] - points2[samples], axis=0).astype(np.int)
        error = points1 - (points2 + xy)
        inlier = np.count_nonzero(np.linalg.norm(error, axis=1) < r)
        if inlier > best_inlier:
            best_inlier = inlier
            best_xy = xy
    
    return best_xy, best_inlier

def intersection(panorama, img, x, y, w, h):
    w_panorama = (panorama > 0).astype(np.float32)
    w_img = np.zeros(panorama.shape)
    w_img[y : y + h, x : x + w][img > 0] = 1
    
    inter = np.bitwise_and(w_panorama.astype(np.bool), w_img.astype(np.bool))
    return inter, w_panorama, w_img

def blending_weight(inter, w_panorama, w_img, panorama, img_p, sign):
    indices = np.nonzero(inter)
    x1 = np.min(indices[1])
    x2 = np.max(indices[1])

    w = np.linspace(0, 1, x2 - x1 + 1) if sign else np.linspace(1, 0, x2 - x1 + 1)
    w = w[np.newaxis, :, np.newaxis]

    w_p = np.copy(w_panorama)
    w_p[:, x1 : x2 + 1] *= (1. - w)
    w_panorama = np.where(inter, w_p, w_panorama)
    
    w_i = np.copy(w_img)
    w_i[:, x1 : x2 + 1] *= w
    w_img = np.where(inter, w_i, w_img)
    return w_panorama, w_img

def _linear_blending(panorama, img, x, y, sign):
    h, w, c = img.shape
    if np.sum(panorama) == 0:
        panorama[y : y + h, x : x + w] = img
    
    else:
        inter, w_panorama, w_img = intersection(panorama, img, x, y, w, h)
        img_p = np.zeros(panorama.shape)
        img_p[y : y + h, x : x + w] = img
        w_panorama, w_img = blending_weight(inter, w_panorama, w_img, panorama, img_p, sign)
        panorama = w_panorama * panorama + w_img * img_p
    
    return panorama

def linear_blending(cyl_img, transform):
    h, w, c = cyl_img[0].shape
    init_pos = np.zeros(2).astype(np.int)
    
    pos = [init_pos]
    for xy in transform:
        pos.append(pos[-1] + xy)
    
    pos = np.array(pos)
    if np.min(pos[:, 0]) < 0:
        pos[:, 0] -= np.min(pos[:, 0])
    if np.min(pos[:, 1]) < 0:
        pos[:, 1] -= np.min(pos[:, 1])

    transform = [init_pos] + transform
    panorama = np.zeros((np.max(pos[:, 1]) + h, np.max(pos[:, 0]) + w, c))
    for img, p, xy in tqdm(zip(cyl_img, pos, transform), total=len(cyl_img)):
        panorama = _linear_blending(panorama, img, p[0], p[1], xy[0] > 0)
    return panorama

def getPanorama(cyl_img, desc_sift):
    print('Find matches & Get best transform')
    transform = []
    for i in range(len(cyl_img) - 1):
        matches = find_matches(desc_sift[i], desc_sift[i + 1])
        kp1 = np.hstack((desc_sift[i]['x'].reshape(-1, 1), desc_sift[i]['y'].reshape(-1, 1)))
        kp2 = np.hstack((desc_sift[i + 1]['x'].reshape(-1, 1), desc_sift[i + 1]['y'].reshape(-1, 1)))

        best_xy, best_inlier = ransac(matches, kp1, kp2)
        print(f'[{i+1:02d} -> {i:02d}] xy: ({best_xy[0]:4d}, {best_xy[1]:4d}), matches: {len(matches):4d}, inlier: {best_inlier:3d}')
        transform.append(best_xy)

    print('\nLinear blending')
    panorama = linear_blending(cyl_img, transform)
    return panorama

if __name__ == '__main__':
    # python3 stitching.py --input_dir ../lake --output_dir ../lake --desc_path ../desc/desc_lake.npy
    # python3 stitching.py --input_dir ../mountain --output_dir ../mountain --desc_path ../desc/desc_mountain.npy
    parser = argparse.ArgumentParser(description='Image Stitching.')
    parser.add_argument('--input_dir', help='Path to the input image dir.')
    parser.add_argument('--output_dir', help='Path to the output image dir.')
    parser.add_argument('--desc_path', help='Path to the desc.npy.')
    args = parser.parse_args()

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cyl_img = []
    for img_path in sorted(glob.glob(args.input_dir + '/cyl*')):
        cyl_img.append(cv2.imread(img_path))

    desc_sift = np.load(args.desc_path, allow_pickle=True)
    panorama = getPanorama(cyl_img, desc_sift)
    cv2.imwrite(f'{args.output_dir}/panorama.jpg', panorama)