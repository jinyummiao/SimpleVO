import torch
import cv2
import numpy as np

from .SuperGlue.models.superglue import SuperGlue


class SuperGlue_Matcher(object):
    def __init__(self, cfg,
        weight_path='src/matcher/SuperGlue/models/weights/superglue_outdoor.pth',
        sinkhorn_iterations=20,
        match_threshold=0.4):
        self.model = SuperGlue({
            'weight_path': weight_path, 
            'sinkhorn_iterations': sinkhorn_iterations, 
            'match_threshold': match_threshold
        })

    def match(self, info1, info2):
        data = {}
        gray0 = cv2.cvtColor(info1['origin'], cv2.COLOR_RGB2GRAY)
        data['image0'] = torch.from_numpy(gray0/255.).float()[None, None]
        data['keypoints0'] = info1['torch']['keypoints'][0].unsqueeze(0)
        data['scores0'] = info1['torch']['scores'][0].unsqueeze(0)
        data['descriptors0'] = info1['torch']['descriptors'][0].unsqueeze(0)

        gray1 = cv2.cvtColor(info2['origin'], cv2.COLOR_RGB2GRAY)
        data['image1'] = torch.from_numpy(gray1/255.).float()[None, None]
        data['keypoints1'] = info2['torch']['keypoints'][0].unsqueeze(0)
        data['scores1'] = info2['torch']['scores'][0].unsqueeze(0)
        data['descriptors1'] = info2['torch']['descriptors'][0].unsqueeze(0)

        pred = self.model(data)

        matches = pred['matches0'][0].cpu().numpy()
        confidence = pred['matching_scores0'][0].cpu().detach().numpy()

        # Sort them in the order of their confidence.
        match_conf = []
        for i, (m, c) in enumerate(zip(matches, confidence)):
            match_conf.append([i, m, c])
        match_conf = sorted(match_conf, key=lambda x: x[2], reverse=True)

        valid = [[l[0], l[1]] for l in match_conf if l[1] > -1]
        
        return np.array(valid, dtype='int')