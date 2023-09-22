import os
import glob
import numpy as np
from scipy.io import loadmat


def main():
    root_data = './data'
    frame_folders = os.listdir(root_data)
    frame_folders.sort()
    for frame_folder in frame_folders:
        root_frame = os.path.join(root_data, frame_folder)

        points_gt = np.load(glob.glob(os.path.join(root_frame, 'gt_frame*.npy'))[0])
        points_gt = points_gt.astype('float')
        if points_gt.shape[0] == 0:
            points_gt = np.zeros((0, 3))

        points_detect_xyz = loadmat(glob.glob(os.path.join(root_frame, 'xyz*float.mat'))[0])['xyz_float']
        points_detect_snr = loadmat(glob.glob(os.path.join(root_frame, 'snr*float.mat'))[0])['snr_float'].T
        points_detect = np.hstack((points_detect_xyz, points_detect_snr))
        points_detect = points_detect.astype('float')
        if points_detect.shape[0] == 0:
            points_detect = np.zeros((0, 4))

        path_output = os.path.join(root_frame, 'data.npz')
        np.savez(path_output, points_gt=points_gt, points_detect=points_detect)



if __name__ == '__main__':
    main()

