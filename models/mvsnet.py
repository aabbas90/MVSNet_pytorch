import torch
import torch.nn as nn
import torch.nn.functional as F
from .module import *
import numpy as np
import scipy.interpolate
from scipy import ndimage
import matplotlib.pyplot as plt
import pdb

class FeatureNet(nn.Module):
    def __init__(self):
        super(FeatureNet, self).__init__()
        self.inplanes = 32

        self.conv0 = ConvBnReLU(3, 8, 3, 1, 1)
        self.conv1 = ConvBnReLU(8, 8, 3, 1, 1)

        self.conv2 = ConvBnReLU(8, 16, 5, 2, 2)
        self.conv3 = ConvBnReLU(16, 16, 3, 1, 1)
        self.conv4 = ConvBnReLU(16, 16, 3, 1, 1)

        self.conv5 = ConvBnReLU(16, 32, 5, 2, 2)
        self.conv6 = ConvBnReLU(32, 32, 3, 1, 1)
        self.feature = nn.Conv2d(32, 32, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(self.conv0(x))
        x = self.conv4(self.conv3(self.conv2(x)))
        x = self.feature(self.conv6(self.conv5(x)))
        return x


class CostRegNet(nn.Module):
    def __init__(self):
        super(CostRegNet, self).__init__()
        self.conv0 = ConvBnReLU3D(32, 8)

        self.conv1 = ConvBnReLU3D(8, 16, stride=2)
        self.conv2 = ConvBnReLU3D(16, 16)

        self.conv3 = ConvBnReLU3D(16, 32, stride=2)
        self.conv4 = ConvBnReLU3D(32, 32)

        self.conv5 = ConvBnReLU3D(32, 64, stride=2)
        self.conv6 = ConvBnReLU3D(64, 64)

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(16, 8, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True))

        self.prob = nn.Conv3d(8, 1, 3, stride=1, padding=1)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        x = conv2 + self.conv9(x)
        x = conv0 + self.conv11(x)
        x = self.prob(x)
        return x


class RefineNet(nn.Module):
    def __init__(self):
        super(RefineNet, self).__init__()
        self.conv1 = ConvBnReLU(4, 32)
        self.conv2 = ConvBnReLU(32, 32)
        self.conv3 = ConvBnReLU(32, 32)
        self.res = ConvBnReLU(32, 1)

    def forward(self, img, depth_init):
        concat = F.cat((img, depth_init), dim=1)
        depth_residual = self.res(self.conv3(self.conv2(self.conv1(concat))))
        depth_refined = depth_init + depth_residual
        return depth_refined


class MVSNet(nn.Module):
    def __init__(self, refine=True):
        super(MVSNet, self).__init__()
        self.refine = refine

        self.feature = FeatureNet()
        self.cost_regularization = CostRegNet()
        if self.refine:
            self.refine_network = RefineNet()

    def forward(self, imgs, proj_matrices, depth_values):
        imgs = torch.unbind(imgs, 1)
        proj_matrices = torch.unbind(proj_matrices, 1)
        assert len(imgs) == len(proj_matrices), "Different number of images and projection matrices"
        img_height, img_width = imgs[0].shape[2], imgs[0].shape[3]
        num_depth = depth_values.shape[1]
        num_views = len(imgs)

        # step 1. feature extraction
        # in: images; out: 32-channel feature maps
        features = [self.feature(img) for img in imgs]
        ref_feature, src_features = features[0], features[1:]
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]

        # step 2. differentiable homograph, build cost volume
        ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, num_depth, 1, 1)
        volume_sum = ref_volume
        volume_sq_sum = ref_volume ** 2
        del ref_volume
        for src_fea, src_proj in zip(src_features, src_projs):
            # warpped features
            warped_volume = homo_warping(src_fea, src_proj, ref_proj, depth_values)
            if self.training:
                volume_sum = volume_sum + warped_volume
                volume_sq_sum = volume_sq_sum + warped_volume ** 2
            else:
                # TODO: this is only a temporal solution to save memory, better way?
                volume_sum += warped_volume
                volume_sq_sum += warped_volume.pow_(2)  # the memory of warped_volume has been modified
            del warped_volume
        # aggregate multiple feature volumes by variance
        volume_variance = volume_sq_sum.div_(num_views).sub_(volume_sum.div_(num_views).pow_(2))

        # step 3. cost volume regularization
        cost_reg = self.cost_regularization(volume_variance)
        # cost_reg = F.upsample(cost_reg, [num_depth * 4, img_height, img_width], mode='trilinear')
        cost_reg = cost_reg.squeeze(1)
        prob_volume = F.softmax(cost_reg, dim=1)
        depth = depth_regression(prob_volume, depth_values=depth_values)

        with torch.no_grad():
            # photometric confidence
            prob_volume_sum4 = 4 * F.avg_pool3d(F.pad(prob_volume.unsqueeze(1), pad=(0, 0, 0, 0, 1, 2)), (4, 1, 1), stride=1, padding=0).squeeze(1)
            depth_index = depth_regression(prob_volume, depth_values=torch.arange(num_depth, device=prob_volume.device, dtype=torch.float)).long()
            photometric_confidence = torch.gather(prob_volume_sum4, 1, depth_index.unsqueeze(1)).squeeze(1)

        # step 4. depth map refinement
        if not self.refine:
            return {"depth": depth, "photometric_confidence": photometric_confidence}
        else:
            refined_depth = self.refine_network(torch.cat((imgs[0], depth), 1))
            return {"depth": depth, "refined_depth": refined_depth, "photometric_confidence": photometric_confidence}


def mvsnet_loss(depth_est, depth_gt, mask):
    mask = mask > 0.5
    return F.smooth_l1_loss(depth_est[mask], depth_gt[mask], size_average=True)

def PixelCoordToWorldCoord(K, R, t, u, v, depth):
    a = np.multiply(K[2,0], u) - K[0,0]
    b = np.multiply(K[2,1], u) - K[0,1]
    c = np.multiply(depth, (K[0,2] - np.multiply(K[2,2], u)))

    g = np.multiply(K[2,0], v) - K[0,1]
    h = np.multiply(K[2,1], v) - K[1,1]
    l = np.multiply(depth, (K[1,2] - np.multiply(K[2,2], v)))

    y = np.divide(l - np.divide(np.multiply(g, c), a), h - np.divide(np.multiply(g, b), a))
    x = np.divide((c - np.multiply(b,y)), a)
    z = depth
    C = np.concatenate(([x], [y], [z]), axis = 0)
    # W = np.matmul(R.T, C - t[:,:,np.newaxis])
    
    W1 = np.sum(R[:,0:1,np.newaxis] * (C - t[:,:,np.newaxis]), axis = 0)
    W2 = np.sum(R[:,1:2,np.newaxis] * (C - t[:,:,np.newaxis]), axis = 0)
    W3 = np.sum(R[:,2:3,np.newaxis] * (C - t[:,:,np.newaxis]), axis = 0)
    W = np.stack((W1, W2, W3), axis = 0)
    return W

def WorldCoordTopixelCoord(K, R, t, W):
    # C = t + np.matmul(R, W.T)
    R1 = R[0:1, :].T
    R2 = R[1:2, :].T
    R3 = R[2:3, :].T
    
    C1 = np.sum(R1[:,:,np.newaxis] * W, axis = 0)
    C2 = np.sum(R2[:,:,np.newaxis] * W, axis = 0)
    C3 = np.sum(R3[:,:,np.newaxis] * W, axis = 0)
    C = t[:,:,np.newaxis] + np.stack((C1, C2, C3), axis = 0)
    # p = np.matmul(K, C)
    K1 = K[0:1, :].T
    K2 = K[1:2, :].T
    K3 = K[2:3, :].T
    
    P1 = np.sum(K1[:,:,np.newaxis] * C, axis = 0)
    P2 = np.sum(K2[:,:,np.newaxis] * C, axis = 0)
    P3 = np.sum(K3[:,:,np.newaxis] * C, axis = 0)
 
    Xp = np.divide(P1, P3)
    Yp = np.divide(P2, P3)
    return Xp, Yp

def GetImageAtPixelCoordinates(image, xp, yp):
    pr = ndimage.map_coordinates(np.squeeze(image[0,:,:]), [yp, xp], order=1)
    pg = ndimage.map_coordinates(np.squeeze(image[1,:,:]), [yp, xp], order=1)
    pb = ndimage.map_coordinates(np.squeeze(image[2,:,:]), [yp, xp], order=1)
    return np.stack((pr, pg, pb), axis = 0)