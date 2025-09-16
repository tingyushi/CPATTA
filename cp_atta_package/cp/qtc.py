import numpy as np
from cp_atta_package.cp.conformal_predictor_base import ConformalPredictor
from cp_atta_package.utils.register import register

from munch import Munch


class MyCustomError(Exception):
    pass


@register.cp_predictor_register
class QTC(ConformalPredictor):
    
    def __init__(self, config):
        self.alpha = config.cp.alpha
        self.num_steps = config.cp.num_steps
        self.T = config.cp.T

        self.cal_scores = None
        self.cal_labels = None
        
        self.beta_qtc_s = None
        self.beta_qtc_t = None
        self.beta_qtc = None
        self.tau_qtc = None
        
    
    def __q_function(self, smx, c):
        max_smx = np.max(smx, axis=1)
        res = np.quantile(max_smx, c, method="linear")
        return res
    
    def __get_beta_QTC(self, source_smx, target_smx, ):
        
        assert source_smx.shape[1] == target_smx.shape[1]
        
        source_size, _ = source_smx.shape
        target_size, _  = target_smx.shape
    
        # get beta QTC-T
        source_max_smx = np.max(source_smx, axis=1)
        q_function_value = self.__q_function(target_smx, self.alpha)
        beta_qtc_t = np.sum( source_max_smx < q_function_value ) / source_size

        # get beta QTC-S
        target_max_smx = np.max(target_smx, axis=1)
        q_function_value = self.__q_function(source_smx, 1 - self.alpha)
        beta_qtc_s = 1 - ( np.sum( target_max_smx < q_function_value ) / target_size )
        
        self.beta_qtc_t = beta_qtc_t
        self.beta_qtc_s = beta_qtc_s
        
        if beta_qtc_s <= beta_qtc_t:
            self.beta_qtc = beta_qtc_s
        else:
            self.beta_qtc = beta_qtc_t
    
    def __C_TPS(self, smx, tau):
        temp = smx >= (1-tau)
        pred_sets = [np.where(temp[i])[0].tolist() for i in range(temp.shape[0])]
        return pred_sets
    
    def calibrate(self, cal_scores, cal_labels, val_scores, verbose=False):
        
        self.cal_scores = cal_scores
        self.cal_labels = cal_labels
        
        # calculate beta_qtc first
        self.__get_beta_QTC(source_smx=cal_scores, target_smx=val_scores)
        assert self.beta_qtc is not None
        
        thr = (1 - self.beta_qtc) * (len(cal_labels) + 1)
        if verbose: print(thr)
        taus = np.linspace(0, 1,  self.num_steps)
        taus = taus[::-1]
        
        # if we cannnot find a valid tau value, we try the best we can
        pred_sets = self.__C_TPS(self.cal_scores, taus.max())
        set_size = sum(1 if d2 in d1 else 0 for d1, d2 in zip(pred_sets, self.cal_labels))
        if set_size < thr: 
            self.tau_qtc = taus.max()
            return
      
        for idx, tau in enumerate(taus):
            pred_sets = self.__C_TPS(self.cal_scores, tau)
            set_size = sum(1 if d2 in d1 else 0 for d1, d2 in zip(pred_sets, self.cal_labels))
            if verbose: print(f"tau = {tau} -- set size = {set_size}")
            if set_size < thr: break  
        
        self.tau_qtc = taus[idx - 1]

    def predict(self, val_scores):
        pred_sets = self.__C_TPS(val_scores, self.tau_qtc)
        
        # generate smooth scores
        sigmoid_input = (val_scores - (1 - self.tau_qtc)) / self.T
        smooth_score = 1 / (1 + np.exp(-1 * sigmoid_input))

        return pred_sets, smooth_score

    def get_key_values(self):
        values = {"beta_qtc": self.beta_qtc, 
                  "beta_qtc_s": self.beta_qtc_s, 
                  "beta_qtc_t": self.beta_qtc_t, 
                  "tau_qtc": self.tau_qtc}
        return Munch(values)
    

    def get_include_larger_than(self):
        assert self.tau_qtc is not None
        return (1 - self.tau_qtc)

    def get_include_smaller_than(self):
        raise MyCustomError("In QTC, this method CANNOT be used!!")