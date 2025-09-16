from typing import List
from numpy.typing import NDArray
from munch import Munch
import copy
from abc import ABC, abstractmethod


class ConformalPredictor(ABC):
    
    @abstractmethod
    def calibrate(self):
        pass    
    
    @abstractmethod
    def predict(self):
        pass
    
    @abstractmethod
    def get_include_larger_than(self):
        pass

    @abstractmethod
    def get_include_smaller_than(self):
        pass

    def evaluate(self, pred_sets: List[List], val_labels: NDArray):
        
        correct = 0
        size = 0
        empty_counter = 0

        for pred_set , val_label in zip(pred_sets, val_labels):
            if val_label in pred_set:
                correct += 1
            if not pred_set:
                empty_counter += 1
            size += len(pred_set)

        cov = correct / len(pred_sets)
        avg_prediction_set_size = size / len(pred_sets)

        res = {"cov": cov,
               "pred_set_size": avg_prediction_set_size, 
               "num_empty_sets": empty_counter}

        return Munch(res)
    
