import cv2


class BF_Matcher(object):
    def __init__(self, cfg):
        self.ratio_test = cfg.MATCHER.RATIO_TEST
        if cfg.MATCHER.NORM.lower() == 'hamming':
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        else:
            self.matcher = cv2.BFMatcher(cv2.NORM_L2)

    def match(self, info1, info2):
        desc1 = info1['opencv']['descriptors']
        desc2 = info2['opencv']['descriptors']
        matches = self.matcher.knnMatch(desc1, desc2, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < self.ratio_test * n.distance:
                good_matches.append([m.queryIdx, m.trainIdx])
        return good_matches
