import glob
import cv2
import tqdm
import numpy as np
import argparse

from config.default_configs import cfg

from src.detector import detector_list

from src.matcher import matcher_list

from src.frame import Frame

from src.utils import calculate_relative_motion, triangulate, plot_tracked_features, get_absolute_scale

from src.map import Map


"""
    Tracking demo
"""
parser = argparse.ArgumentParser(description='Tracking demo.')
parser.add_argument('--config',
                    type=str, 
                    default='config/params/kitti_sift_bf.yaml',
                    help='configuraton of experiments')
args = parser.parse_args()

if args.config in ['config/params/kitti_orb_bf.yaml', 'config/params/kitti_sift_bf.yaml', 'config/params/kitti_superpoint_nn.yaml', 'config/params/kitti_superpoint_superglue.yaml']:
    # load configuration
    cfg.merge_from_file(args.config)
    cfg.freeze()
    print(cfg)
else:
    raise TypeError('No such configuration!')

# load_config_file = 'config/params/kitti_orb_bf.yaml' # ORB detector + Brute-Force matcher
# load_config_file = 'config/params/kitti_sift_bf.yaml' # SIFT detector + Brute-Force matcher
# load_config_file = 'config/params/kitti_superpoint_nn.yaml' # SuperPoint detector + Mutual-NN matcher
# load_config_file = 'config/params/kitti_superpoint_superglue.yaml' # SuperPoint detector + SuperGlue matcher


# init
imgs = sorted(glob.glob(cfg.TEST.IMG_ROOT + cfg.TEST.IMG_TYPE))
with open(cfg.TEST.GT_POSE, 'r') as f:
    text = f.readlines()
gt_poses = np.zeros((len(text), 4, 4))
for i, line in enumerate(text):
    words = line.split('\n')[0].split(' ')
    gt_poses[i] = np.array([[eval(words[0]), eval(words[1]), eval(words[2]), eval(words[3])],
                            [eval(words[4]), eval(words[5]), eval(words[6]), eval(words[7])],
                            [eval(words[8]), eval(words[9]), eval(words[10]), eval(words[11])],
                            [0, 0, 0, 1]])
feat = detector_list[cfg.FEATURE.NAME.lower()](cfg.FEATURE.NUMS)
matcher = matcher_list[cfg.MATCHER.NAME.lower()](cfg)
mapp = Map(1024, 640)
if cfg.TEST.SHOW_TRACKING:
    win = 'Tracking demo'
    cv2.namedWindow(win)

# run
idx = 0
for img_path in tqdm.tqdm(imgs):
    cur_info = feat.detect(img_path)

    frame = Frame(cfg.TEST.KEYFRAME)
    frame.info = cur_info
    frame.idx = idx

    if frame.idx == 0:
        # first frame
        frame.pose = gt_poses[idx]
        frame.gt = gt_poses[idx]
        frame.keyframe = True
        points4d = [[0, 0, 0, 1]] # set original point (0,0,0), 1 indicates the color
    else:
        # track with former frame
        matches = matcher.match(frame.info, last_keyframe.info)
        if len(matches) <= cfg.TEST.KEYFRAME_INLIERS: # select a new keyframe
            frame.keyframe = True
        assert len(matches) >= 8 # if not, tracking is lost

        R, t = calculate_relative_motion(cfg, matches, frame.info, last_keyframe.info) # Tlc
        frame.gt = gt_poses[idx]

        scale = get_absolute_scale(frame, last_keyframe)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3:] = t * scale

        frame.pose = np.dot(last_keyframe.pose, T) # Twc = Twl * Tlc

        # triangulation
        points4d = triangulate(cfg, matches, frame.info, last_keyframe.info, frame.pose, last_keyframe.pose)

        # show tracking
        if cfg.TEST.SHOW_TRACKING:
            board = plot_tracked_features(frame, last_keyframe, matches)
            cv2.imshow(win, board)
        
    mapp.add_observation(frame.pose, frame.gt, points4d) # add the current camera and landmarks into the map
    mapp.display()

    if frame.keyframe:
        last_keyframe = frame
    last_frame = frame
    idx += 1

    key = cv2.waitKey(10) & 0xFF

    if key == ord('q'):
        print('Quitting.')
        break

cv2.destroyAllWindows()
print('Demo ends.')
