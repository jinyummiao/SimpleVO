import cv2
import numpy as np


class SIFT_Feat(object):

    def __init__(self, chosen_num):
        self.chosen_num = chosen_num
        self.detector = cv2.SIFT_create(self.chosen_num,
                                        nOctaveLayers=3,
                                        contrastThreshold=0.04,
                                        edgeThreshold=10,
                                        sigma=1.6)


    def detect(self, img_path):
        img0 = cv2.imread(img_path)
        gray = cv2.cvtColor(img0, cv2.COLOR_RGB2GRAY)
        kpt = self.detector.detect(gray)
        kpt, des = self.detector.compute(gray, kpt)

        np_kpt = np.array([[p.pt[0], p.pt[1]] for p in kpt])
        np_des = np.array(des)


        return {'origin':img0,
            'opencv':
                {'keypoints': kpt,
                'descriptors': des},
            'numpy':
                {'keypoints': np_kpt,
                'descriptors':np_des}
        }


