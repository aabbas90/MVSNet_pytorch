import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt
from datasets.data_io import read_pfm, save_pfm
import pdb

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('depth_path')
    args = parser.parse_args()
    depth_path = args.depth_path
    if depth_path.endswith('npy'):
        depth_image = np.load(depth_path)
        depth_image = np.squeeze(depth_image)
        print('value range: ', depth_image.min(), depth_image.max())
        plt.imshow(depth_image, 'rainbow')
        plt.show()
    elif depth_path.endswith('pfm'):
        depth_image = read_pfm(depth_path)
        ma = np.ma.masked_equal(depth_image[0], 0.0, copy=False)
        print('value range: ', ma.min(), ma.max())
        plt.imshow(depth_image[0], 'rainbow')
        plt.show()
    else:
        depth_image = cv2.imread(depth_path)
        ma = np.ma.masked_equal(depth_image, 0.0, copy=False)
        print('value range: ', ma.min(), ma.max())
        plt.imshow(depth_image)
        plt.show()
