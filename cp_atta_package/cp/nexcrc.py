import numpy as np
from numpy.typing import NDArray
from cp_atta_package.cp.conformal_predictor_base import ConformalPredictor
from cp_atta_package.utils.register import register

class MyCustomError(Exception):
    pass


@register.nexcrc_loss_func_register
def coverage_loss(prediction_sets, true_labels):
    assert prediction_sets.dtype == np.bool_
    assert prediction_sets.shape[0] == len(true_labels)
    rows = np.arange(len(true_labels))
    correct_class_mask = prediction_sets[rows, true_labels]
    loss_array = np.where(correct_class_mask, 0, 1)
    return loss_array


@register.cp_predictor_register
class DescentFasterNexCRC(ConformalPredictor):
    
    def __init__(self, config):

        self.alpha = config.cp.alpha
        self.T = config.cp.T
        self.weights = None
        self.loss_func_upper_bound = config.cp.loss_func_upper_bound
        self.num_steps = config.cp.num_steps
        self.loss_func = register.nexcrc_loss_funcs[config.cp.loss_func_name]
        
        self.cal_scores = None
        self.cal_labels = None
        self.lambda_hat = None

        # because we need a compensate value, so the actual include
        self.include_larger_than_value = None
        
    def calibrate(self, cal_scores: NDArray, cal_labels: NDArray, verbose=False):
        
        self.cal_scores = cal_scores
        self.cal_labels = cal_labels
        
        # assert number of weights are the same as number of calibration data
        assert self.weights is not None
        assert len(self.weights) == self.cal_scores.shape[0]

        # set up a set of lambdas, and then we gonna choose lambda hat from here
        lambdas = np.linspace(0, 1, self.num_steps) # probability threshold
        
        # reverse lambdas for faster computation
        lambdas = lambdas[::-1]
        
        # compute n_w, the weights for all the calibratoin data samples
        n_w = np.sum(self.weights)

        if verbose:
            print(f"Include labels in the prediction set only if their softmax scores are equal/larger than (1-lambda)")
        
        # creare two lists to record the process
        metric_scores_records = []
        lambda_value_records = []
        
        for idx, lambda_ in enumerate(lambdas):

            # if the prob >= 1-lambda_, then include this in the prediction set
            prediction_sets = self.cal_scores >= (1 - lambda_)
            
            # calculate losses for this lambda
            losses = self.loss_func(prediction_sets, self.cal_labels)

            # calculate R_hat for each lambda value
            r_hat = (1/n_w) * np.sum( self.weights * losses )
            
            # calculate metric score
            metric_score = (r_hat * n_w / (n_w+1) ) + self.loss_func_upper_bound / (n_w + 1)
            
            # record process
            metric_scores_records.append(metric_score)
            lambda_value_records.append(lambda_)
            
            if (idx == 0) and (metric_score > self.alpha):
                if verbose:
                    print(f"The Smallest F(lambda) = {metric_score} is even larger alpha = {self.alpha}, so the best is to let lambda_hat = 1")
                lambda_chosen = 1
                break
            
            if metric_score > self.alpha:
                lambda_chosen = lambdas[idx - 1]
                break
            
            if (idx == (len(lambdas) - 1)) :
                lambda_chosen = 0
                break


        if verbose:
            for d1, d2 in zip(metric_scores_records, lambda_value_records):
                print(f"F(lambda) = {d1:.4f} when lambda = {d2:.4f}")
            
            print(f"So we include labels if their softmax score is equal/larger than {1 - lambda_chosen}")
        
        self.lambda_hat = lambda_chosen

        self.weights = None
        
    
    def predict(self, val_scores, compensate=0):
        assert self.lambda_hat is not None

        # calculate include larger than value
        include_larger_than_value = 1 - self.lambda_hat

        # consider compensate value
        include_larger_than_value += compensate
        include_larger_than_value = max(include_larger_than_value, 0)  # ensure it's not negative

        # assign value
        self.include_larger_than_value = include_larger_than_value

        # generate prediction set
        prediction_sets = val_scores >= self.include_larger_than_value
        pred_sets = [np.where(prediction_sets[i])[0].tolist() for i in range(prediction_sets.shape[0])]

        # generate smooth scores
        sigmoid_input = (val_scores - self.include_larger_than_value) / self.T
        smooth_score = 1 / (1 + np.exp(-1 * sigmoid_input))

        return pred_sets, smooth_score
    

    def update_weights(self, new_weights):
        self.weights = new_weights

    def get_lambda_hat(self):
        assert self.lambda_hat is not None
        return self.lambda_hat

    def get_include_larger_than(self):
        assert self.include_larger_than_value is not None
        return self.include_larger_than_value

    def get_include_smaller_than(self):
        raise MyCustomError("In NEXCRC, this method CANNOT be used!!")



@register.cp_predictor_register
class AscentFasterNexCRC(ConformalPredictor):
    
    def __init__(self, config):
        
        self.alpha = config.cp.alpha
        self.T = config.cp.T
        self.weights = None
        self.loss_func_upper_bound = config.cp.loss_func_upper_bound
        self.num_steps = config.cp.num_steps
        self.loss_func = register.nexcrc_loss_funcs[config.cp.loss_func_name]
        
        self.cal_scores = None
        self.cal_labels = None
        self.lambda_hat = None

        # because we need a compensate value, so the actual include
        self.include_larger_than_value = None
        
        
    def calibrate(self, cal_scores: NDArray, cal_labels: NDArray, verbose=False):
        
        self.cal_scores = cal_scores
        self.cal_labels = cal_labels
        
        # assert number of weights are the same as number of calibration data
        assert self.weights is not None
        assert len(self.weights) == self.cal_scores.shape[0]

        # set up a set of lambdas, and then we gonna choose lambda hat from here
        lambdas = np.linspace(0, 1, self.num_steps) # probability threshold
         
        # compute n_w, the weights for all the calibratoin data samples
        n_w = np.sum(self.weights)

        if verbose:
            print(f"Include labels in the prediction set only if their softmax scores are equal/larger than (1-lambda)")
        
        # creare two lists to record the process
        metric_scores_records = []
        lambda_value_records = []
        
        for idx, lambda_ in enumerate(lambdas):

            # if the prob >= 1-lambda_, then include this in the prediction set
            prediction_sets = self.cal_scores >= (1 - lambda_)

            # calculate losses for this lambda
            losses = self.loss_func(prediction_sets, self.cal_labels)

            # calculate R_hat for each lambda value
            r_hat = (1/n_w) * np.sum( self.weights * losses )
            
            # calculate metric score
            metric_score = (r_hat * n_w / (n_w+1) ) + self.loss_func_upper_bound / (n_w + 1)
            
            # record process
            metric_scores_records.append(metric_score)
            lambda_value_records.append(lambda_)
            
            if (idx ==  (len(lambdas) - 1) ) and (metric_score > self.alpha):
                if verbose:
                    print(f"The Smallest F(lambda) = {metric_score} is even larger alpha = {self.alpha}, so the best is to let lambda_hat = 1")
                lambda_chosen = 1
                break
            
            if metric_score <= self.alpha:
                lambda_chosen = lambdas[idx]
                break
            
        
        if verbose:
            for d1, d2 in zip(metric_scores_records, lambda_value_records):
                print(f"F(lambda) = {d1:.4f} when lambda = {d2:.4f}")
            
            print(f"So we include labels if their softmax score is equal/larger than {1 - lambda_chosen}")
        
        self.lambda_hat = lambda_chosen
        
        self.weights = None

    
    def predict(self, val_scores, compensate=0):
        assert self.lambda_hat is not None

        # calculate include larger than value
        include_larger_than_value = 1 - self.lambda_hat

        # consider compensate value
        include_larger_than_value += compensate
        include_larger_than_value = max(include_larger_than_value, 0)  # ensure it's not negative

        # assign value
        self.include_larger_than_value = include_larger_than_value

        # generate prediction sets
        prediction_sets = val_scores >= self.include_larger_than_value
        pred_sets = [np.where(prediction_sets[i])[0].tolist() for i in range(prediction_sets.shape[0])]

        # generate smooth scores
        sigmoid_input = (val_scores - self.include_larger_than_value) / self.T
        smooth_score = 1 / (1 + np.exp(-1 * sigmoid_input))

        return pred_sets, smooth_score
    
    def get_lambda_hat(self):
        assert self.lambda_hat is not None
        return self.lambda_hat

    def update_weights(self, new_weights):
        self.weights = new_weights

    def get_include_larger_than(self):
        assert self.include_larger_than_value is not None
        return self.include_larger_than_value

    def get_include_smaller_than(self):
        raise MyCustomError("In NEXCRC, this method CANNOT be used!!")