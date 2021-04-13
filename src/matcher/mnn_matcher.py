import numpy as np


class MNN_Matcher(object):
    def __init__(self, cfg):
        self.nn_thresh = cfg.MATCHER.NN_THRESHOLD
        if self.nn_thresh < 0.0:
            raise ValueError('\'nn_thresh\' should be non-negative')

    def match(self, info1, info2):
        desc1 = info1['numpy']['descriptors']
        desc2 = info2['numpy']['descriptors']
        assert desc1.shape[1] == desc2.shape[1]
        if desc1.shape[0] == 0 or desc2.shape[0] == 0:
            return np.zeros((3, 0))

        dmat = np.dot(desc1, desc2.T)
        dmat = np.sqrt(2 - 2 * np.clip(dmat, -1, 1))

        # Get NN indices and scores.
        idx = np.argmin(dmat, axis=1)
        scores = dmat[np.arange(dmat.shape[0]), idx]
        # Threshold the NN matches.
        keep = scores < self.nn_thresh
        # Check if nearest neighbor goes both directions and keep those.
        idx2 = np.argmin(dmat, axis=0)
        keep_bi = np.arange(len(idx)) == idx2[idx]
        keep = np.logical_and(keep, keep_bi)
        idx = idx[keep]
        scores = scores[keep]
        # Get the surviving point indices.
        m_idx1 = np.arange(desc1.shape[0])[keep]
        m_idx2 = idx
        # Populate the final Nx2 match data structure.
        matches = np.zeros((2, int(keep.sum())))
        matches[0, :] = m_idx1
        matches[1, :] = m_idx2
        # matches[2, :] = scores
        matches = np.array(matches, dtype='int')
        return matches.T
