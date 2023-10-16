import cv2
import glob
import pathlib
import argparse
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # python3 crc.py --crc ../images/crc.npy
    parser = argparse.ArgumentParser()
    parser.add_argument('--crc', help='Path to the camera response curve.')
    args = parser.parse_args()
    crc = np.load(args.crc).astype(np.float32)
    
    plt.figure()
    plt.title('Camera response curve')
    plt.plot(crc[0], range(256), color='b', marker='.', markersize=1)
    plt.plot(crc[1], range(256), color='g', marker='.', markersize=1)
    plt.plot(crc[2], range(256), color='r', marker='.', markersize=1)
    plt.xlabel('log exposure X')
    plt.ylabel('pixel value Z')
    plt.tight_layout()
    plt.savefig('../images/crc.jpg')
    plt.close()