from .orb_detector import ORB_Feat
from .sift_detector import SIFT_Feat
from .superpoint_detector_v2 import SuperPoint_Feat

detector_list = {
    'orb': ORB_Feat,
    'sift': SIFT_Feat,
    'superpoint': SuperPoint_Feat
}