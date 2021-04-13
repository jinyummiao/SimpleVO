from yacs.config import CfgNode as CN

_CN = CN()

# image resolution
_CN.IMG = CN()
_CN.IMG.H = 640
_CN.IMG.W = 480

# chosed features
_CN.FEATURE = CN()
_CN.FEATURE.NUMS = 1000
_CN.FEATURE.NAME = 'ORB' # SIFT, SuperPoint

# matcher
_CN.MATCHER = CN()
_CN.MATCHER.NAME = 'BF' # MNN, SuperGlue(can be used with SuperPoint only)
_CN.MATCHER.NORM = 'hamming'
_CN.MATCHER.RATIO_TEST = 1.0
_CN.MATCHER.NN_THRESHOLD = 0.7

# camera
_CN.CAMERA = CN()
_CN.CAMERA.FOCAL = 0.0
_CN.CAMERA.PP = (0.0, 0.0)

# run
_CN.TEST = CN()
_CN.TEST.IMG_ROOT = '/media/mjy/jinyu.miao/kitti05/'
_CN.TEST.GT_POSE = '/media/mjy/jinyu.miao/05.txt'
_CN.TEST.IMG_TYPE = '*.png'
_CN.TEST.SHOW_TRACKING = True
_CN.TEST.KEYFRAME = True
_CN.TEST.KEYFRAME_INLIERS = 50

def get_cfg_defaults():
	return _CN.clone()

cfg = _CN
