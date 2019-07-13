from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
import cv2
import pdb
from datasets.data_io import *


# the DTU dataset preprocessed by Yao Yao (only for testing)
class MVSDataset(Dataset):
    def __init__(self, datapath, listfile, mode, nviews, ndepths=128, interval_scale=1.06, max_dim = 768, **kwargs):
        super(MVSDataset, self).__init__()
        self.datapath = datapath
        self.listfile = listfile
        self.mode = mode
        self.nviews = nviews
        self.ndepths = ndepths
        self.interval_scale = interval_scale
        self.max_dim = max_dim

        assert self.mode == "test"
        self.metas = self.build_list()

    def build_list(self):
        metas = []
        with open(self.listfile) as f:
            scans = f.readlines()
            scans = [line.rstrip() for line in scans]

        # scans
        for scan in scans:
            pair_file = "{}/pair.txt".format(scan)
            # read the pair file
            with open(os.path.join(self.datapath, pair_file)) as f:
                num_viewpoint = int(f.readline())
                # viewpoints (49)
                for view_idx in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    metas.append((scan, ref_view, src_views))
        print("dataset", self.mode, "metas:", len(metas))
        return metas

    def __len__(self):
        return len(self.metas)

    def read_cam_file(self, filename, scaling_w, scaling_h):
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
        intrinsics[0, :] *= scaling_w
        intrinsics[1, :] *= scaling_h
        # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0])
        depth_interval = float(lines[11].split()[1]) * self.interval_scale
        return intrinsics, extrinsics, depth_min, depth_interval

    def read_img(self, filename):
        img = Image.open(filename)
        # scale 0~255 to 0~1
        np_img = np.array(img, dtype=np.float32) / 255.
        # assert np_img.shape[:2] == (1200, 1600)
        # crop to (1184, 1600)
        # np_img = np_img[:-16, :]  # do not need to modify intrinsics if cropping the bottom part
        return np_img

        
    def resize_img(self, img, w, h):
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

    def determine_new_h_w(self, orig_w, orig_h):
        orig_max_dim = np.max((orig_w, orig_h))
        scale_factor = self.max_dim / orig_max_dim.astype(float) 
        new_w = int(np.round((orig_w * scale_factor) / 32.0) * 32)
        new_h = int(np.round((orig_h * scale_factor) / 32.0) * 32)
        scale_w = new_w / orig_w
        scale_h = new_h / orig_h 
        return new_w, new_h, scale_w, scale_h

    def read_depth(self, filename):
        # read pfm depth file
        return np.array(read_pfm(filename)[0], dtype=np.float32)

    def __getitem__(self, idx):
        meta = self.metas[idx]
        scan, ref_view, src_views = meta
        # use only the reference view and first nviews-1 source views
        view_ids = [ref_view] + src_views[:self.nviews - 1]

        imgs = []
        mask = None
        depth = None
        depth_values = None
        proj_matrices = []
        int_matrices = []
        ext_matrices = []
        new_w = None

        for i, vid in enumerate(view_ids):
            img_filename = os.path.join(self.datapath, '{}/images/{:0>8}.jpg'.format(scan, vid))
            proj_mat_filename = os.path.join(self.datapath, '{}/cams/{:0>8}_cam.txt'.format(scan, vid))

            orig_image = self.read_img(img_filename)
            if new_w is None:
                new_w, new_h, scale_w, scale_h = self.determine_new_h_w(orig_image.shape[1], orig_image.shape[0])

            resized_image = self.resize_img(orig_image, new_w, new_h)
            imgs.append(resized_image)
            intrinsics, extrinsics, depth_min, depth_interval = self.read_cam_file(proj_mat_filename, scale_w * 0.25, scale_h * 0.25)

            # multiply intrinsics and extrinsics to get projection matrix
            proj_mat = extrinsics.copy()
            proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])
            proj_matrices.append(proj_mat)
            int_matrices.append(intrinsics)
            ext_matrices.append(extrinsics)

            if i == 0:  # reference view
                depth_values = np.arange(depth_min, depth_interval * (self.ndepths - 0.5) + depth_min, depth_interval,
                                         dtype=np.float32)

        imgs = np.stack(imgs).transpose([0, 3, 1, 2])
        proj_matrices = np.stack(proj_matrices)

        return {"imgs": imgs,
                "proj_matrices": proj_matrices,
                "depth_values": depth_values,
                "filename": scan + '/{}/' + '{:0>8}'.format(view_ids[0]) + "{}",
                "int_matrices": int_matrices,
                "ext_matrices": ext_matrices}


if __name__ == "__main__":
    # some testing code, just IGNORE it
    dataset = MVSDataset("/home/xyguo/dataset/dtu_mvs/processed/mvs_testing/dtu/", '../lists/dtu/test.txt', 'test', 5,
                         128)
    item = dataset[50]
    for key, value in item.items():
        print(key, type(value))
