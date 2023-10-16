import cv2
import glob
import rawpy
import pathlib
import argparse
import numpy as np
from tqdm import tqdm

def get_bitmap(img: np.ndarray, ignore: int=4):
    median = np.median(img)
    ret, mtb = cv2.threshold(img, median, 255, cv2.THRESH_BINARY)
    exclusion = np.zeros(img.shape).astype(np.bool)
    exclusion[np.where(img < median - ignore)] = 1
    exclusion[np.where(img > median + ignore)] = 1
    return mtb, exclusion

def shift_img(img: np.ndarray, x: int, y: int):
    M = np.float32([
        [1, 0, x],
        [0, 1, y],
    ])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

def MTB(img1: np.ndarray, img2: np.ndarray, level: int):
    shift_x = shift_y = 0
    if level > 0:
        img1_shrink = cv2.resize(img1, (0, 0), fx=0.5, fy=0.5)
        img2_shrink = cv2.resize(img2, (0, 0), fx=0.5, fy=0.5)
        shift_x, shift_y = MTB(img1_shrink, img2_shrink, level=level-1)
        shift_x *= 2
        shift_y *= 2

    mtb1, exclusion1 = get_bitmap(img1)
    mtb2, exclusion2 = get_bitmap(img2)

    best_shift_x = best_shift_y = 0
    best_err = img1.shape[0] * img1.shape[1]
    for x in range(-1, 2):
        for y in range(-1, 2):
            cur_x = shift_x + x
            cur_y = shift_y + y
            shifted_mtb2 = shift_img(mtb2, cur_x, cur_y)

            diff = np.bitwise_xor(mtb1, shifted_mtb2)
            diff = np.bitwise_and(diff, exclusion1)
            err = np.count_nonzero(diff)

            if err < best_err:
                best_err = err
                best_shift_x = cur_x
                best_shift_y = cur_y

    return best_shift_x, best_shift_y

def align_img(gray_img: list):
    if len(gray_img) <= 1:
        return gray_img
    
    shift = [(0, 0)]
    for i in tqdm(range(1, len(gray_img))):
        shift_x, shift_y = MTB(gray_img[0], gray_img[i], level=5)
        shift.append((shift_x, shift_y))
    return shift

if __name__ == '__main__':
    # python3 mtb.py --input_dir ../data --output_dir ../mtb
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', help='Path to the input image dir.')
    parser.add_argument('--output_dir', help='Path to the output image dir.')
    args = parser.parse_args()
    
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gray_dir = pathlib.Path('../gray')
    gray_dir.mkdir(parents=True, exist_ok=True)

    gray_img, bgr_img = [], []
    for img_path in sorted(glob.glob(args.input_dir + '/*.jpg')):
        # # Convert raw image to numpy RGB array
        # raw = rawpy.imread(img_path)
        # rgb = raw.postprocess()

        bgr = cv2.imread(img_path)
        gray_img.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
        bgr_img.append(bgr)
        # Save gray images
        gray_path = gray_dir / img_path.split('/')[-1]
        cv2.imwrite(str(gray_path), gray_img[-1])

    shift = align_img(gray_img)
    for i, ((shift_x, shift_y), img_path) in enumerate(zip(shift, sorted(glob.glob(args.input_dir + '/*jpg')))):
        print(f'Shift ({shift_x}, {shift_y}) pixels for {img_path}')
        output_path = output_dir / img_path.split('/')[-1]
        cv2.imwrite(str(output_path), shift_img(bgr_img[i], shift_x, shift_y))