import numpy as np
import torch
import cv2

from .SuperPoint.demo_superpoint import SuperPointFrontend


class SuperPoint_Feat(object):

    def __init__(self, chosen_num,
        weights_path='src/detector/SuperPoint/superpoint_v1.pth', 
        nms_dist=4, 
        conf_thresh=0.015, 
        cuda=torch.cuda.is_available()):
        self.chosen_num = chosen_num
        self.model = SuperPointFrontend(weights_path, nms_dist, conf_thresh, cuda)
    
    def detect(self, img_path):
        img0 = cv2.imread(img_path)
        gray0 = cv2.cvtColor(img0, cv2.COLOR_RGB2GRAY)
        input_image = gray0.astype('float')/255.0
        input_image = input_image.astype('float32')
        kpts, desc, scores = self.model.run(input_image)
        kpts = kpts.T 
        desc = desc.T 
        assert len(kpts) == len(desc)

        kpts = np.array(sorted(kpts, key=lambda x: x in kpts[:, 2], reverse=True))
        desc = np.array(sorted(desc, key=lambda x: x in kpts[:, 2], reverse=True))

        if kpts.shape[0] > self.chosen_num:
            kpts = kpts[:self.chosen_num, :]
            desc = desc[:self.chosen_num, :]

        kpts = kpts[:, :2]
        kpts = np.asarray(kpts, dtype='float')

        return {
            'origin': img0,
            'numpy': {
                'keypoints': kpts,
                'descriptors': desc
            }
        }