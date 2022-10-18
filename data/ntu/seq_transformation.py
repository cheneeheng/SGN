# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import os
import os.path as osp
import numpy as np
import pickle
import logging
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from typing import Tuple

import math


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    if np.abs(axis).sum() < 1e-6 or np.abs(theta) < 1e-6:
        return np.eye(3)
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    if np.abs(v1).sum() < 1e-6 or np.abs(v2).sum() < 1e-6:
        return 0
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def x_rotation(vector, theta):
    """Rotates 3-D vector around x-axis"""
    R = np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])
    return np.dot(R, vector)


def y_rotation(vector, theta):
    """Rotates 3-D vector around y-axis"""
    R = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
    return np.dot(R, vector)


def z_rotation(vector, theta):
    """Rotates 3-D vector around z-axis"""
    R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    return np.dot(R, vector)
    

def __verbose(x, out=None, verbose=False):
    if verbose:
        print(out)
        return tqdm(x)
    else:
        return x


def pre_normalization(data,
                      zaxis=[0, 1],
                      zaxis2=None,
                      xaxis=[8, 4],
                      pad=True,
                      center=True,
                      center_firstframe=False,
                      verbose=False,
                      tqdm=True):

    # data : (num_skes, max_num_frames, 150)

    if center or center_firstframe:
        assert center_firstframe != center

    N, T, MVC = data.shape
    data = data.reshape(N, T, 2, -1, 3)  # n,t,m,v,c

    s = np.transpose(data, [0, 2, 1, 3, 4])  # N, M, T, V, C

    if verbose:
        for i_s, skeleton in enumerate(s):
            if skeleton.sum() == 0:
                print(f'Seq {i_s} has no skeleton')

    if zaxis is not None:
        s_list = __verbose(s, out='parallel the bone between hip(jpt 0) and spine(jpt 1) of the first person to the z axis', verbose=verbose)  # noqa
        for i_s, skeleton in enumerate(s_list):
            if skeleton.sum() == 0:
                continue
            joint_bottom = skeleton[0, 0, zaxis[0]]
            joint_top = skeleton[0, 0, zaxis[1]]
            axis = np.cross(joint_top - joint_bottom, [0, 0, 1])
            angle = angle_between(joint_top - joint_bottom, [0, 0, 1])
            matrix_z = rotation_matrix(axis, angle)
            # print(f"z angle {angle}")
            for i_p, person in enumerate(skeleton):
                if person.sum() == 0:
                    continue
                for i_f, frame in enumerate(person):
                    if frame.sum() == 0:
                        continue
                    for i_j, joint in enumerate(frame):
                        s[i_s, i_p, i_f, i_j] = np.dot(matrix_z, joint)

    if xaxis is not None:
        s_list = __verbose(s, out='parallel the bone between right shoulder(jpt 8) and left shoulder(jpt 4) of the first person to the x axis', verbose=verbose)  # noqa
        for i_s, skeleton in enumerate(s_list):
            if skeleton.sum() == 0:
                continue
            joint_rshoulder = skeleton[0, 0, xaxis[0]]
            joint_lshoulder = skeleton[0, 0, xaxis[1]]
            axis = np.cross(joint_rshoulder - joint_lshoulder, [1, 0, 0])
            angle = angle_between(joint_rshoulder - joint_lshoulder, [1, 0, 0])
            matrix_x = rotation_matrix(axis, angle)
            # print(f"x angle {angle}")
            for i_p, person in enumerate(skeleton):
                if person.sum() == 0:
                    continue
                for i_f, frame in enumerate(person):
                    if frame.sum() == 0:
                        continue
                    for i_j, joint in enumerate(frame):
                        s[i_s, i_p, i_f, i_j] = np.dot(matrix_x, joint)

    s = np.transpose(data, [0, 2, 1, 3, 4])  # N, T, M, V, C
    data = data.reshape(N, T, MVC)

    return data


# WHATS HAPPENING HERE:
# - load raw data
# - translate sequence to the real first frame
# - real first frame is defined by the first non zero frame of actior 1
# - align/pad the frames with zeros until max number (300)
# - split datasets

def remove_nan_frames(ske_name: str,
                      ske_joints: np.ndarray,
                      nan_logger: logging.Logger) -> np.ndarray:
    num_frames = ske_joints.shape[0]
    valid_frames = []

    for f in range(num_frames):
        if not np.any(np.isnan(ske_joints[f])):
            valid_frames.append(f)
        else:
            nan_indices = np.where(np.isnan(ske_joints[f]))[0]
            nan_logger.info('{}\t{:^5}\t{}'.format(
                ske_name, f + 1, nan_indices))

    return ske_joints[valid_frames]


def seq_translation(skes_joints: list) -> list:
    for idx, ske_joints in enumerate(tqdm(skes_joints)):
        num_frames = ske_joints.shape[0]
        num_bodies = 1 if ske_joints.shape[1] == 75 else 2
        if num_bodies == 2:
            missing_frames_1 = np.where(ske_joints[:, :75].sum(axis=1) == 0)[0]
            missing_frames_2 = np.where(ske_joints[:, 75:].sum(axis=1) == 0)[0]
            cnt1 = len(missing_frames_1)
            cnt2 = len(missing_frames_2)

        i = 0  # get the "real" first frame of actor1
        while i < num_frames:
            if np.any(ske_joints[i, :75] != 0):
                break
            i += 1

        # new origin: joint-2
        origin = np.copy(ske_joints[i, joint_2[0]:joint_2[1]])

        for f in range(num_frames):
            if num_bodies == 1:
                ske_joints[f] -= np.tile(origin, 25)
            else:  # for 2 actors
                ske_joints[f] -= np.tile(origin, 50)

        # makes sure the zero frames stay zero frames
        if (num_bodies == 2) and (cnt1 > 0):
            ske_joints[missing_frames_1, :75] = np.zeros(
                (cnt1, 75), dtype=np.float32)

        if (num_bodies == 2) and (cnt2 > 0):
            ske_joints[missing_frames_2, 75:] = np.zeros(
                (cnt2, 75), dtype=np.float32)

        skes_joints[idx] = ske_joints  # Update

    return skes_joints


def frame_translation(skes_joints: list,
                      skes_name: str,
                      frames_cnt: np.ndarray) -> Tuple[list, np.ndarray]:
    nan_logger = logging.getLogger('nan_skes')
    nan_logger.setLevel(logging.INFO)
    nan_logger.addHandler(logging.FileHandler("./nan_frames.log"))
    nan_logger.info('{}\t{}\t{}'.format('Skeleton', 'Frame', 'Joints'))

    for idx, ske_joints in enumerate(skes_joints):
        num_frames = ske_joints.shape[0]
        # Calculate the distance between
        # spine base (joint-1) and spine (joint-21)
        j1 = ske_joints[:, joint_1[0]:joint_1[1]]
        j21 = ske_joints[:, joint_21[0]:joint_21[1]]
        dist = np.sqrt(((j1 - j21) ** 2).sum(axis=1))

        for f in range(num_frames):
            # new origin: middle of the spine (joint-2)
            origin = ske_joints[f, joint_2[0]:joint_2[1]]
            if (ske_joints[f, 75:] == 0).all():
                ske_joints[f, :75] = (
                    ske_joints[f, :75] - np.tile(origin, 25)) / \
                    dist[f] + np.tile(origin, 25)
            else:
                ske_joints[f] = (ske_joints[f] - np.tile(origin, 50)
                                 ) / dist[f] + np.tile(origin, 50)

        ske_name = skes_name[idx]
        ske_joints = remove_nan_frames(ske_name, ske_joints, nan_logger)
        frames_cnt[idx] = num_frames  # update valid number of frames
        skes_joints[idx] = ske_joints

    return skes_joints, frames_cnt


def align_frames(skes_joints: list, frames_cnt: np.ndarray) -> np.ndarray:
    """
    Align all sequences with the same frame length.

    """
    num_skes = len(skes_joints)
    max_num_frames = frames_cnt.max()  # 300
    aligned_skes_joints = np.zeros(
        (num_skes, max_num_frames, 150), dtype=np.float32)

    for idx, ske_joints in enumerate(tqdm(skes_joints)):
        num_frames = ske_joints.shape[0]
        num_bodies = 1 if ske_joints.shape[1] == 75 else 2
        if num_bodies == 1:
            aligned_skes_joints[idx, :num_frames] = np.hstack(
                (ske_joints, np.zeros_like(ske_joints)))
        else:
            aligned_skes_joints[idx, :num_frames] = ske_joints

    return aligned_skes_joints


# def one_hot_vector(labels: np.ndarray) -> np.ndarray:
#     num_skes = len(labels)
#     labels_vector = np.zeros((num_skes, 60))
#     for idx, l in enumerate(labels):
#         labels_vector[idx, l] = 1

#     return labels_vector


def split_train_val(train_indices: np.ndarray,
                    method: str = 'sklearn',
                    ratio: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get validation set by splitting data randomly from training set
    with two methods.
    In fact, I thought these two methods are equal as they got the
    same performance.

    """
    if method == 'sklearn':
        return train_test_split(train_indices, test_size=ratio,
                                random_state=10000)
    else:
        np.random.seed(10000)
        np.random.shuffle(train_indices)
        val_num_skes = int(np.ceil(0.05 * len(train_indices)))
        val_indices = train_indices[:val_num_skes]
        train_indices = train_indices[val_num_skes:]
        return train_indices, val_indices


def split_dataset(skes_joints: list,
                  label: np.ndarray,
                  performer: np.ndarray,
                  camera: np.ndarray,
                  evaluation: str,
                  save_path: str):
    train_indices, test_indices = get_indices(performer, camera, evaluation)
    m = 'sklearn'  # 'sklearn' or 'numpy'
    # Select validation set from training set
    train_indices, val_indices = split_train_val(train_indices, m)

    # Save labels and num_frames for each sequence of each data set
    train_labels = label[train_indices]
    val_labels = label[val_indices]
    test_labels = label[test_indices]

    with open(osp.join(save_path, f'NTU_{evaluation}_train.pkl'), 'wb') as fw:
        pickle.dump(skes_joints[train_indices], fw, pickle.HIGHEST_PROTOCOL)
    with open(osp.join(save_path, f'NTU_{evaluation}_train_label.pkl'), 'wb') as fw:  # noqa
        pickle.dump(train_labels, fw, pickle.HIGHEST_PROTOCOL)

    with open(osp.join(save_path, f'NTU_{evaluation}_val.pkl'), 'wb') as fw:
        pickle.dump(skes_joints[val_indices], fw, pickle.HIGHEST_PROTOCOL)
    with open(osp.join(save_path, f'NTU_{evaluation}_val_label.pkl'), 'wb') as fw:  # noqa
        pickle.dump(val_labels, fw, pickle.HIGHEST_PROTOCOL)

    with open(osp.join(save_path, f'NTU_{evaluation}_test.pkl'), 'wb') as fw:
        pickle.dump(skes_joints[test_indices], fw, pickle.HIGHEST_PROTOCOL)
    with open(osp.join(save_path, f'NTU_{evaluation}_test_label.pkl'), 'wb') as fw:  # noqa
        pickle.dump(test_labels, fw, pickle.HIGHEST_PROTOCOL)

    # # Save data into a .h5 file
    # h5file = h5py.File(osp.join(save_path, 'NTU_%s.h5' % (evaluation)), 'w')
    # # Training set
    # h5file.create_dataset('x', data=skes_joints[train_indices])
    # train_one_hot_labels = one_hot_vector(train_labels)
    # h5file.create_dataset('y', data=train_one_hot_labels)
    # # Validation set
    # h5file.create_dataset('valid_x', data=skes_joints[val_indices])
    # val_one_hot_labels = one_hot_vector(val_labels)
    # h5file.create_dataset('valid_y', data=val_one_hot_labels)
    # # Test set
    # h5file.create_dataset('test_x', data=skes_joints[test_indices])
    # test_one_hot_labels = one_hot_vector(test_labels)
    # h5file.create_dataset('test_y', data=test_one_hot_labels)

    # h5file.close()


def get_indices(performer: np.ndarray,
                camera: np.ndarray,
                evaluation: str = 'CS') -> Tuple[np.ndarray, np.ndarray]:
    test_indices = np.empty(0)
    train_indices = np.empty(0)

    if evaluation == 'CS':  # Cross Subject (Subject IDs)
        train_ids = [1,  2,  4,  5,  8,  9,  13, 14, 15, 16,
                     17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
        test_ids = [3,  6,  7,  10, 11, 12, 20, 21, 22, 23,
                    24, 26, 29, 30, 32, 33, 36, 37, 39, 40]

        # Get indices of test data
        for idx in test_ids:
            temp = np.where(performer == idx)[0]  # 0-based index
            test_indices = np.hstack((test_indices, temp)).astype(int)

        # Get indices of training data
        for train_id in train_ids:
            temp = np.where(performer == train_id)[0]  # 0-based index
            train_indices = np.hstack((train_indices, temp)).astype(int)
    else:  # Cross View (Camera IDs)
        train_ids = [2, 3]
        test_ids = 1
        # Get indices of test data
        temp = np.where(camera == test_ids)[0]  # 0-based index
        test_indices = np.hstack((test_indices, temp)).astype(int)

        # Get indices of training data
        for train_id in train_ids:
            temp = np.where(camera == train_id)[0]  # 0-based index
            train_indices = np.hstack((train_indices, temp)).astype(int)

    return train_indices, test_indices


if __name__ == '__main__':

    joint_1 = (0, 3)  # spine base
    joint_2 = (3, 6)  # middle of the spine
    joint_21 = (60, 63)  # spine

    root_path = './data/ntu'
    stat_path = osp.join(root_path, 'statistics')
    camera_file = osp.join(stat_path, 'camera.txt')
    performer_file = osp.join(stat_path, 'performer.txt')
    label_file = osp.join(stat_path, 'label.txt')

    skes_name_file = osp.join(stat_path, 'skes_available_name.txt')

    root_path = './_data/data/ntu_sgn'
    denoised_path = osp.join(root_path, 'denoised_data')
    raw_skes_joints_pkl = osp.join(denoised_path, 'raw_denoised_joints.pkl')
    frames_file = osp.join(denoised_path, 'frames_cnt.txt')

    save_path = osp.join(root_path, 'processed_data')

    if not osp.exists(save_path):
        os.mkdir(save_path)

    camera = np.loadtxt(camera_file, dtype=int)  # camera id: 1, 2, 3
    performer = np.loadtxt(performer_file, dtype=int)  # subject id: 1~40
    label = np.loadtxt(label_file, dtype=int) - 1  # action label: 0~59

    frames_cnt = np.loadtxt(frames_file, dtype=int)  # frames_cnt
    skes_name = np.loadtxt(skes_name_file, dtype=np.string_)

    with open(raw_skes_joints_pkl, 'rb') as fr:
        skes_joints = pickle.load(fr)  # a list

    print(f"Translating seq")
    skes_joints = seq_translation(skes_joints)

    # skes_joints, frames_cnt = frame_translation(skes_joints, skes_name,
    #                                             frames_cnt)

    print(f"Aligning seq")
    # aligned to the same frame length
    skes_joints = align_frames(skes_joints, frames_cnt)

    print(f"AAGCN PRE-NORM")
    skes_joints = pre_normalization(skes_joints,
                                    zaxis=[0, 1],
                                    zaxis2=None,
                                    xaxis=[8, 4],
                                    pad=False,
                                    center=False,
                                    center_firstframe=False,
                                    verbose=False,
                                    tqdm=True)

    evaluations = ['CS', 'CV']
    for evaluation in evaluations:
        print(f"Evaluating {evaluation}")
        split_dataset(skes_joints, label, performer,
                      camera, evaluation, save_path)
