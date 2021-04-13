import numpy as np


class Frame(object):
    def __init__(self, keyframe=False):
        self.idx = 0
        self.keyframe = not keyframe

        self.info = None
        self.pose = np.eye(4) # camera pose in world coordinate system, Pw.
        self.gt = np.eye(4)
