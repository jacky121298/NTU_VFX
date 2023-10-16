import cv2
import glob
import random
import pathlib
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def harris_corner(gray, ksize=3, sigma=3, k=0.04):
    gray_blur = cv2.GaussianBlur(gray, (ksize, ksize), sigma)
    Iy, Ix = np.gradient(gray_blur)

    Ix2 = Ix ** 2
    Iy2 = Iy ** 2
    Ixy = Ix * Iy

    Sx2 = cv2.GaussianBlur(Ix2, (ksize, ksize), sigma)
    Sy2 = cv2.GaussianBlur(Iy2, (ksize, ksize), sigma)
    Sxy = cv2.GaussianBlur(Ixy, (ksize, ksize), sigma)

    detM = (Sx2 * Sy2) - (Sxy ** 2)
    traceM = Sx2 + Sy2
    R = detM - k * (traceM ** 2)
    return R, Ix, Iy, Ix2, Iy2

def nms(R, ksize=3, q=99.99):
    kernel = []
    for r in range(ksize):
        for c in range(ksize):
            if r == 1 and c == 1:
                continue
            
            k = np.zeros((ksize, ksize))
            k[ksize // 2, ksize // 2] = 1
            k[r, c] = -1
            kernel.append(k)

    local_max = np.ones(R.shape).astype(np.uint8)
    local_max[R <= np.percentile(R, q=q)] = 0

    for k in kernel:
        s = np.sign(cv2.filter2D(R, ddepth=-1, kernel=k)).astype(np.uint8)
        s[s < 0] = 0
        local_max = np.bitwise_and(local_max, s)

    corners = np.nonzero(local_max > 0)
    return corners[1], corners[0]

def save_corners(img, corners_x, corners_y, i):
    harris_img = np.copy(img)
    for x, y in zip(corners_x, corners_y):
        harris_img = cv2.circle(harris_img, (x, y), radius=10, color=[0, 0, 255], thickness=-1)
    cv2.imwrite(f'{args.output_dir}/harris_{i:02d}.jpg', harris_img)

def get_descriptor(gray_img, corners_x, corners_y):
    desc = {'x': [], 'y': [], 'd': []}
    for (x, y) in zip(corners_x, corners_y):
        desc['x'].append(x)
        desc['y'].append(y)
        
        d = []
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                d.append(gray_img[y + dy, x + dx])
        desc['d'].append(np.array(d))

    desc['x'] = np.array(desc['x'])
    desc['y'] = np.array(desc['y'])
    desc['d'] = np.array(desc['d'])
    return desc

def get_orientations(Ix, Iy, Ix2, Iy2, bins=8, ksize=9):
    M = (Ix2 + Iy2) ** (1/2)
    theta = np.arctan(Iy / (Ix + 1e-8)) * (180 / np.pi)
    theta[Ix < 0] += 180
    theta = (theta + 360) % 360
    
    bin_size = 360. / bins
    theta_bins = (theta + (bin_size / 2)) // int(bin_size) % bins
    ori_1hot = np.zeros((bins,) + Ix.shape)
    
    for b in range(bins):
        ori_1hot[b][theta_bins == b] = 1
        ori_1hot[b] *= M
        ori_1hot[b] = cv2.GaussianBlur(ori_1hot[b], (ksize, ksize), 0)
    
    ori = np.argmax(ori_1hot, axis=0)
    return ori, ori_1hot, theta, theta_bins, M

def get_descriptors(fpx, fpy, ori, theta):
    bins, h, w = ori.shape
    
    def get_sub_vector(fx, fy, ox, oy, ori):
        sv = []
        for b in range(bins):
            sv.append(np.sum(ori[b][fy : fy+oy, fx : fx+ox]))
        sv_n1 = [x / (np.sum(sv) + 1e-8) for x in sv]
        sv_clip = [x if x < 0.2 else 0.2 for x in sv_n1]
        sv_n2 = [x / (np.sum(sv_clip) + 1e-8) for x in sv_clip]
        return sv_n2
    
    def get_vector(x, y):
        M = cv2.getRotationMatrix2D((12, 12), theta[y, x], 1)
        if y-12 < 0 or x-12 < 0: return 0
        ori_rotated = [cv2.warpAffine(t[y-12 : y+12, x-12 : x+12], M, (24, 24)) for t in ori]
        vector = []
        subpatch_offsets = [4, 8, 12, 16]
        for fy in subpatch_offsets:
            for fx in subpatch_offsets:
                vector += get_sub_vector(fx, fy, 4, 4, ori_rotated)
        return vector
    
    desc = {'x': [], 'y': [], 'd': []}
    for x, y in zip(fpx, fpy):
        vector = get_vector(x, y)
        if np.sum(vector) > 0:
            desc['x'].append(x)
            desc['y'].append(y)
            desc['d'].append(vector)

    desc['x'] = np.array(desc['x'])
    desc['y'] = np.array(desc['y'])
    desc['d'] = np.array(desc['d'])
    return desc

if __name__ == '__main__':
    # python3 harris.py --input_dir ../lake --output_dir ../lake --desc_path ../desc/desc_lake.npy
    # python3 harris.py --input_dir ../mountain --output_dir ../mountain --desc_path ../desc/desc_mountain.npy
    parser = argparse.ArgumentParser(description='Harris corner detection & Get descriptor.')
    parser.add_argument('--input_dir', help='Path to the input image dir.')
    parser.add_argument('--output_dir', help='Path to the output image dir.')
    parser.add_argument('--desc_path', help='Path to the desc.npy.')
    args = parser.parse_args()

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cyl_img = []
    for img_path in sorted(glob.glob(args.input_dir + '/cyl*')):
        cyl_img.append(cv2.imread(img_path))

    print('Harris corner detection & Get descriptors')
    desc_sift = []
    for i in tqdm(range(len(cyl_img))):
        gray_img = cv2.cvtColor(cyl_img[i], cv2.COLOR_BGR2GRAY)
        R, Ix, Iy, Ix2, Iy2 = harris_corner(gray_img)
        corners_x, corners_y = nms(R)
        save_corners(cyl_img[i], corners_x, corners_y, i)

        ori, ori_1hot, theta, theta_bins, M = get_orientations(Ix, Iy, Ix2, Iy2)
        desc = get_descriptors(corners_x, corners_y, ori_1hot, theta)
        desc_sift.append(desc)
    
    print('Save descriptors')
    np.save(args.desc_path, desc_sift)