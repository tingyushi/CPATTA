import numpy as np
from numpy.typing import NDArray
from cp_atta_package.cp.conformal_predictor_base import ConformalPredictor
import torch


class MyCustomError(Exception):
    pass

"""
We use torch for this cp predictor since we want it to be differentiable
"""
class SmoothCP(ConformalPredictor):

    def __init__(self, alpha, score_type, Temp):
        self.cal_scores = None
        self.cal_labels = None
        self.alpha = alpha
        self.score_type = score_type
        self.tau = None
        self.T = Temp

    def calibrate(self, cal_scores: NDArray, cal_labels: NDArray, verbose=False):
        
        self.cal_scores = torch.from_numpy(cal_scores)
        self.cal_labels = torch.from_numpy(cal_labels)
        
        assert self.score_type == "conformality"

        n, _ = self.cal_scores.shape
        conformality_scores = [ self.cal_scores[i][self.cal_labels[i].item()].item() for i in range(n) ]
        quantile_value = self.alpha * (1 + (1/n))
        self.tau = torch.quantile(torch.FloatTensor(conformality_scores), quantile_value)


    def predict(self, val_scores: NDArray):
        val_scores = torch.from_numpy(val_scores)
        sigmoid_input = (val_scores - self.tau) / self.T
        smooth_cp_scores = torch.sigmoid( sigmoid_input )
        return smooth_cp_scores

    def get_tau(self):
        return self.tau
    
    def get_include_larger_than(self):
        raise MyCustomError("In Smooth CP, this method CANNOT be used!!")

    def get_include_smaller_than(self):
        raise MyCustomError("In Smooth CP, this method CANNOT be used!!")