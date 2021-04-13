import cv2
import numpy as np


"""
Some function in simple visual odometry implements.
"""
# pixel coordinate to normalized coordinate
def normalize(cfg, pts):
    npts = []
    for p in pts:
        npts.append([(p[0] - cfg.CAMERA.PP[0]) / cfg.CAMERA.FOCAL, (p[1] - cfg.CAMERA.PP[1]) / cfg.CAMERA.FOCAL])
    return np.array(npts)


def calculate_relative_motion(cfg, matches, info1, info2):
    matches = np.array(matches)
    mkpt1 = info1['numpy']['keypoints'][matches[:, 0]]
    mkpt2 = info2['numpy']['keypoints'][matches[:, 1]]
    assert len(mkpt1) == len(mkpt2)

    E_matrix, mask = cv2.findEssentialMat(mkpt1, mkpt2,
                                    focal=cfg.CAMERA.FOCAL,
                                    pp=cfg.CAMERA.PP,
                                    method=cv2.RANSAC,
                                    prob=0.999, threshold=1.0)
    _, R, t, mask = cv2.recoverPose(E_matrix, mkpt1, mkpt2,
                                    focal=cfg.CAMERA.FOCAL,
                                    pp=cfg.CAMERA.PP,
                                    )
    return R, t


def get_absolute_scale(frame, last_frame):
    return np.sqrt((frame.gt[0,3]-last_frame.gt[0,3])**2 \
        + (frame.gt[1,3]-last_frame.gt[1,3])**2 \
            + (frame.gt[2,3]-last_frame.gt[2,3])**2)


def triangulate(cfg, matches, info1, info2, pose1, pose2):
    matches = np.array(matches)
    mkpt1 = info1['numpy']['keypoints'][matches[:, 0]]
    mkpt2 = info2['numpy']['keypoints'][matches[:, 1]]
    assert len(mkpt1) == len(mkpt2)

    mkpt1 = normalize(cfg, mkpt1)
    mkpt2 = normalize(cfg, mkpt2)

    pose1 = np.linalg.inv(pose1) # Tcw
    pose2 = np.linalg.inv(pose2)

    points4d = np.zeros((mkpt1.shape[0], 4)) # landmarks
    for i, (kp1, kp2) in enumerate(zip(mkpt1, mkpt2)):
        A = np.zeros((4, 4))
        A[0] = kp1[0] * pose1[2] - pose1[0]
        A[1] = kp1[1] * pose1[2] - pose1[1]
        A[2] = kp2[0] * pose2[2] - pose2[0]
        A[3] = kp2[1] * pose2[2] - pose2[1]
        _, _, vt = np.linalg.svd(A)
        points4d[i] = vt[3]
    points4d /= points4d[:, 3:]
    good_idx = points4d[:, 2] > 0
    return points4d[good_idx]


def plot_tracked_features(frame1, frame2, matches):
    img1 = frame1.info['origin']
    img2 = frame2.info['origin']

    h, w, c = img1.shape

    board = np.concatenate([img1, img2], axis=0)

    matches = np.array(matches)
    mkpt1 = frame1.info['numpy']['keypoints'][matches[:, 0]]
    mkpt2 = frame2.info['numpy']['keypoints'][matches[:, 1]]
    assert len(mkpt1) == len(mkpt2)

    for kp1, kp2 in zip(mkpt1, mkpt2):
        u1, v1 = int(kp1[0]), int(kp1[1])
        u2, v2 = int(kp2[0]), int(kp2[1])
        cv2.circle(board, (u1, v1), color=(0,0,255), radius=5)
        cv2.circle(board, (u2, v2+h), color=(0,0,255), radius=5)
        cv2.line(board, (u1, v1), (u2, v2+h), color=(255,0,0))
        
    return board
    