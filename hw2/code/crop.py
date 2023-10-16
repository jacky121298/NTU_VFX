import cv2
import glob
import random
import imutils
import pathlib
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

if __name__ == '__main__':
    # python3 crop.py --input ../lake/panorama.jpg --output ../lake/panorama_crop.jpg
    # python3 crop.py --input ../mountain/panorama.jpg --output ../mountain/panorama_crop.jpg
    parser = argparse.ArgumentParser(description='Crop panorama to rectangle.')
    parser.add_argument('--input', help='Path to the input image.')
    parser.add_argument('--output', help='Path to the output image.')
    args = parser.parse_args()

    print('Crop panorama')
    panorama = cv2.imread(args.input)
    panorama_crop = cv2.copyMakeBorder(panorama, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0, 0, 0))
    gray = cv2.cvtColor(panorama_crop, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)

    mask = np.zeros(thresh.shape, dtype='uint8')
    (x, y, w, h) = cv2.boundingRect(c)
    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

    minRect = mask.copy()
    sub = mask.copy()
    while cv2.countNonZero(sub) > 0:
        minRect = cv2.erode(minRect, None)
        sub = cv2.subtract(minRect, thresh)

    cnts = cv2.findContours(minRect.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    (x, y, w, h) = cv2.boundingRect(c)

    panorama_crop = panorama_crop[y : y + h, x : x + w]
    cv2.imwrite(args.output, panorama_crop)