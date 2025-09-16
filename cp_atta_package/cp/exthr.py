import numpy as np
from numpy.typing import NDArray
from cp_atta_package.cp.conformal_predictor_base import ConformalPredictor
from cp_atta_package.utils.register import register


@register.cp_predictor_register
class ExTHR(ConformalPredictor):
    
    def __init__(self, config):
        self.alpha = config.cp.alpha
        self.T = config.cp.T

        self.cal_scores = None
        self.cal_labels = None
        self.thr = None

    def calibrate(self, cal_scores: NDArray, cal_labels: NDArray, verbose=False):
        
        self.cal_scores = cal_scores
        self.cal_labels = cal_labels
        
        n, _ = cal_scores.shape
        temp = np.squeeze(cal_scores[np.arange(n), cal_labels])
        q_level = np.ceil((n+1)*(self.alpha))/n
        thr = np.quantile(temp, q_level, method="lower")

        self.thr = thr
        
    def predict(self, val_scores: NDArray):
        
        # generate prediction set
        temp = val_scores >= self.thr
        pred_sets = [np.where(temp[i])[0].tolist() for i in range(temp.shape[0])]

        # generate smooth cp scores
        sigmoid_input = (val_scores - self.thr) / self.T
        smooth_score = 1 / (1 + np.exp(-1 * sigmoid_input))

        return pred_sets, smooth_score
    
    def get_thr(self):
        return self.thr

    def get_include_larger_than(self):
        assert self.thr is not None
        return self.thr

    def get_include_smaller_than(self):
        return super().get_include_smaller_than()