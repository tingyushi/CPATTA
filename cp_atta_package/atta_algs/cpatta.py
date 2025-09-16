from munch import Munch
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from abc import ABC, abstractmethod
import gc
import copy

from cp_atta_package.utils import initialize, model_utils, uncertainty_measure, data_utils, domain_shift_degree_eval
from cp_atta_package.utils.register import register
from cp_atta_package.atta_algs import cpatta_base



"""
Non-Exchangeable Conformal Risk Control
We use two cp predictors to select data

1. For pseudo labeled data: we used pretrained model -- thus we only need to calibrate once if the weights remain the same
2. For oracle labeled data: we used realtime model -- this we need to calibrate for every batch of data
3. select more data when encountering a new domain for the first time
"""
@register.atta_alg_register
class CPATTA_NexCRC(cpatta_base.CPATTA_Base):

    def __init__(self, config):
        super().__init__(config)
        
        # we need two cp predictors
        self.realtime_cp_predictor = register.cp_predictors[config.cp.cp_name](config)
        self.pretrained_cp_predictor = register.cp_predictors[config.cp.cp_name](config)

        # we store two lr values for seperate training
        self.oracle_lr = config.train.oracle_lr
        self.pseudo_lr = config.train.pseudo_lr

        # handle how to use the cp result
        self.use_smooth = config.atta.use_smooth
        self.smooth_topK_avg = config.atta.smooth_topK_avg
        
        # what to use for weights
        self.use_dss = config.cp.use_dss
        self.use_const_weight = config.cp.use_const_weight
        self.use_time_series = config.cp.use_time_series
        self.use_cos_similarity = config.cp.use_cos_similarity
        self.use_pseudo_coverage = config.cp.use_pseudo_coverage

        # assert one use one of the ways to determine weights
        assert sum([self.use_dss, self.use_const_weight, self.use_time_series, self.use_cos_similarity, self.use_pseudo_coverage]) == 1

        self.config = config

        # set up variables for dss, this is used for detecting domain change
        self.dss_k = config.atta.dss_k_value
        self.dss_history = []
        self.previous_feats = None


        # set up variables for selecting more data when encountering a new domain for the first time
        self.oracle_num_new_domain = config.atta.oracle_num_new_domain


        # we also need two sets of cp corresponding variables
        self.realtime_cp_eval_res = None
        self.realtime_cp_include_larger_than = None
        self.realtime_cp_pred_sets = None
        self.realtime_cp_smooth_scores = None
        self.realtime_cp_cal_weights_mean = 0

        self.pretrained_cp_eval_res = None
        self.pretrained_cp_include_larger_than = None
        self.pretrained_cp_pred_sets = None
        self.pretrained_cp_smooth_scores = None
        self.pretrained_cp_cal_weights_mean = 0

        # some variables to avoid mutiple calibrations/inference on pretrained model
        self.pretrained_cp_calibrated = False
        self.pretrained_cal_inference_munch = None
        self.cal_labels = None


        # define some parameters for used for adjusting scalefactor value
        self.pretrained_cal_entropy = None
        self.realtime_cal_entropy = None
        self.pretrained_tta_entropy = None
        self.realtime_tta_entropy = None

        self.pretrained_cp_previous_scalefactor = None
        self.realtime_cp_previous_scalefactor = None
        self.pretrained_cp_previous_weight = None
        self.realtime_cp_previous_weight = None


    def select_data(self, tt_dataloader, cal_dataloader=None):
        
        assert cal_dataloader is not None

        # model inference
        if self.pretrained_cal_inference_munch is None:
            self.pretrained_cal_inference_munch = model_utils.model_inference(self.teacher, cal_dataloader, self.device)
            self.pretrained_cal_entropy = uncertainty_measure.entropy(self.pretrained_cal_inference_munch.numpy_smx).mean_entropy
        realtime_cal_inference_munch = model_utils.model_inference(self.model, cal_dataloader, self.device)
        pretrained_tta_inference_munch = model_utils.model_inference(self.teacher, tt_dataloader, self.device)
        realtime_tta_inference_munch = model_utils.model_inference(self.model, tt_dataloader, self.device)

        # calculate entropy
        self.realtime_cal_entropy = uncertainty_measure.entropy(realtime_cal_inference_munch.numpy_smx).mean_entropy
        self.pretrained_tta_entropy = uncertainty_measure.entropy(pretrained_tta_inference_munch.numpy_smx).mean_entropy
        self.realtime_tta_entropy = uncertainty_measure.entropy(realtime_tta_inference_munch.numpy_smx).mean_entropy


        # store cal_labels
        if self.cal_labels is None:
            self.cal_labels = data_utils.extract_labels(cal_dataloader)
        
        tt_labels = data_utils.extract_labels(tt_dataloader)
        
        torch_true_labels = torch.cat([labels for _, labels in tt_dataloader], dim=0)
        torch_data = torch.cat([d for d, _ in tt_dataloader], dim=0)
        torch_pred_labels = realtime_tta_inference_munch.torch_pred_labels.cpu()
        torch_feats = realtime_tta_inference_munch.torch_feats.cpu()
        # torch_pred_labels_pretrained = pretrained_tta_inference_munch.torch_pred_labels.cpu()


        if self.use_const_weight:
            
            # update weights
            new_weights = np.array( [self.config.cp.const_weight] * len(cal_dataloader.dataset) )
            self.pretrained_cp_predictor.update_weights(new_weights)
            self.realtime_cp_predictor.update_weights(new_weights)


            # store the cp calibration data weights
            self.pretrained_cp_cal_weights_mean = np.mean( self.pretrained_cp_predictor.weights )
            self.realtime_cp_cal_weights_mean = np.mean( self.realtime_cp_predictor.weights )


            # calibrate pretrained cp
            if not self.pretrained_cp_calibrated:
                cal_scores = self.pretrained_cal_inference_munch.numpy_smx
                self.pretrained_cp_predictor.calibrate(cal_scores, self.cal_labels)
                self.pretrained_cp_calibrated = True


            # calibrate realtime cp
            cal_scores = realtime_cal_inference_munch.numpy_smx
            self.realtime_cp_predictor.calibrate(cal_scores, self.cal_labels)


            # pretrained cp prediction
            tt_scores = pretrained_tta_inference_munch.numpy_smx
            pretrained_cp_pred_sets, pretrained_cp_smooth_scores = self.pretrained_cp_predictor.predict(tt_scores)


            # realtime cp prediction
            tt_scores = realtime_tta_inference_munch.numpy_smx
            realtime_cp_pred_sets, realtime_cp_smooth_scores = self.realtime_cp_predictor.predict(tt_scores)
            

            # store include larger than
            self.pretrained_cp_include_larger_than = [self.pretrained_cp_predictor.get_include_larger_than()] * len(tt_dataloader.dataset)
            self.realtime_cp_include_larger_than = [self.realtime_cp_predictor.get_include_larger_than()] * len(tt_dataloader.dataset)

        if self.use_time_series:
        
            # upate weights 
            new_weights = self.config.cp.rho ** (np.arange(len(cal_dataloader.dataset), 0, -1))
            self.pretrained_cp_predictor.update_weights(new_weights)
            self.realtime_cp_predictor.update_weights(new_weights)
            
            
            # store the cp calibration data weights
            self.pretrained_cp_cal_weights_mean = np.mean( self.pretrained_cp_predictor.weights )
            self.realtime_cp_cal_weights_mean = np.mean( self.realtime_cp_predictor.weights )


            # calibrate pretrained cp
            if not self.pretrained_cp_calibrated:
                cal_scores = self.pretrained_cal_inference_munch.numpy_smx
                self.pretrained_cp_predictor.calibrate(cal_scores, self.cal_labels)
                self.pretrained_cp_calibrated = True


            # calibrate realtime cp
            cal_scores = realtime_cal_inference_munch.numpy_smx
            self.realtime_cp_predictor.calibrate(cal_scores, self.cal_labels)


            # pretrained cp prediction
            tt_scores = pretrained_tta_inference_munch.numpy_smx
            pretrained_cp_pred_sets, pretrained_cp_smooth_scores = self.pretrained_cp_predictor.predict(tt_scores)


            # realtime cp prediction
            tt_scores = realtime_tta_inference_munch.numpy_smx
            realtime_cp_pred_sets, realtime_cp_smooth_scores = self.realtime_cp_predictor.predict(tt_scores)
            

            # store include larger than
            self.pretrained_cp_include_larger_than = [self.pretrained_cp_predictor.get_include_larger_than()] * len(tt_dataloader.dataset)
            self.realtime_cp_include_larger_than = [self.realtime_cp_predictor.get_include_larger_than()] * len(tt_dataloader.dataset)

        
        if self.use_cos_similarity:

            # calculate weights
            if self.config.cp.fast_cos:
                realtime_cp_weights_mat =  domain_shift_degree_eval.fast_cosine_similarity(self.model, self.device, cal_dataloader, tt_dataloader)
                pretrained_cp_weights_mat =  domain_shift_degree_eval.fast_cosine_similarity(self.teacher, self.device, cal_dataloader, tt_dataloader)
            else:
                realtime_cp_weights_mat =  domain_shift_degree_eval.slow_cosine_similarity(self.model, self.device, cal_dataloader, tt_dataloader)
                pretrained_cp_weights_mat =  domain_shift_degree_eval.slow_cosine_similarity(self.teacher, self.device, cal_dataloader, tt_dataloader)

            pretrained_cp_weights_mat = (pretrained_cp_weights_mat + 1) / self.config.cp.cos_similarity_T
            realtime_cp_weights_mat = (realtime_cp_weights_mat + 1) / self.config.cp.cos_similarity_T

            # pretrained model prediction
            tt_scores = pretrained_tta_inference_munch.numpy_smx
            weights_mat = pretrained_cp_weights_mat
            cal_scores = self.pretrained_cal_inference_munch.numpy_smx

            if not self.config.cp.use_mean_weights_mat:
                pretrained_cp_pred_sets = []
                pretrained_cp_smooth_scores = []
                self.pretrained_cp_include_larger_than = []
                for idx, tt_score in enumerate(tt_scores):
                    self.pretrained_cp_predictor.update_weights(weights_mat[idx])
                    self.pretrained_cp_predictor.calibrate(cal_scores, self.cal_labels)
                    pred_set, smooth_score = self.pretrained_cp_predictor.predict( np.array([tt_score]) )
                    pretrained_cp_pred_sets.append(pred_set[0])
                    pretrained_cp_smooth_scores.append(smooth_score[0])
                    self.pretrained_cp_include_larger_than.append( self.pretrained_cp_predictor.get_include_larger_than() )
                pretrained_cp_smooth_scores = np.array(pretrained_cp_smooth_scores)
            else:
                weights_mat = np.mean(weights_mat, axis=0)
                self.pretrained_cp_predictor.update_weights(weights_mat)
                self.pretrained_cp_predictor.calibrate(cal_scores, self.cal_labels)
                pretrained_cp_pred_sets, pretrained_cp_smooth_scores = self.pretrained_cp_predictor.predict(tt_scores)
                self.pretrained_cp_include_larger_than = [self.pretrained_cp_predictor.get_include_larger_than()] * len(tt_dataloader.dataset)
                self.pretrained_cp_cal_weights_mean = np.mean(weights_mat)

            # realtime model prediction
            tt_scores = realtime_tta_inference_munch.numpy_smx
            weights_mat = realtime_cp_weights_mat
            cal_scores = model_utils.model_inference(self.model, cal_dataloader, self.device).numpy_smx
            if not self.config.cp.use_mean_weights_mat:
                realtime_cp_pred_sets = []
                realtime_cp_smooth_scores = []
                self.realtime_cp_include_larger_than = []
                for idx, tt_score in enumerate(tt_scores):
                    self.realtime_cp_predictor.update_weights(weights_mat[idx])
                    self.realtime_cp_predictor.calibrate(cal_scores, self.cal_labels)
                    pred_set, smooth_score = self.realtime_cp_predictor.predict( np.array([tt_score]) )
                    realtime_cp_pred_sets.append(pred_set[0])
                    realtime_cp_smooth_scores.append(smooth_score[0])
                    self.realtime_cp_include_larger_than.append( self.realtime_cp_predictor.get_include_larger_than() )
                realtime_cp_smooth_scores = np.array(realtime_cp_smooth_scores)
            else:
                weights_mat = np.mean(weights_mat, axis=0)
                self.realtime_cp_predictor.update_weights(weights_mat)
                self.realtime_cp_predictor.calibrate(cal_scores, self.cal_labels)
                realtime_cp_pred_sets, realtime_cp_smooth_scores = self.realtime_cp_predictor.predict(tt_scores)
                self.realtime_cp_include_larger_than = [self.realtime_cp_predictor.get_include_larger_than()] * len(tt_dataloader.dataset)
                self.realtime_cp_cal_weights_mean = np.mean(weights_mat)

        if self.use_dss:

            # update new weights
            dss_value = domain_shift_degree_eval.domain_shift_signal_using_feats(self.pretrained_cal_inference_munch.numpy_feats, 
                                                                                   pretrained_tta_inference_munch.numpy_feats)
            
            # scale down dss value
            pretrained_weight_single_value = dss_value / self.config.cp.pretrained_cp_dssT
            realtime_weight_single_value = dss_value / self.config.cp.realtime_cp_dssT

            # calculate cp weights
            pretrained_weights = np.array( [ pretrained_weight_single_value ] * len(cal_dataloader.dataset) )
            realtime_weights = np.array( [ realtime_weight_single_value ] * len(cal_dataloader.dataset) )


            # assign weights
            self.pretrained_cp_predictor.update_weights(pretrained_weights)
            self.realtime_cp_predictor.update_weights(realtime_weights)


            # store the cp calibration data weights
            self.pretrained_cp_cal_weights_mean = np.mean( self.pretrained_cp_predictor.weights )
            self.realtime_cp_cal_weights_mean = np.mean( self.realtime_cp_predictor.weights )


            # calibrate pretrained cp
            cal_scores = self.pretrained_cal_inference_munch.numpy_smx
            self.pretrained_cp_predictor.calibrate(cal_scores, self.cal_labels)
            

            # calibrate realtime cp
            cal_scores = realtime_cal_inference_munch.numpy_smx
            self.realtime_cp_predictor.calibrate(cal_scores, self.cal_labels)
            

            # pretrained cp prediction
            tt_scores = pretrained_tta_inference_munch.numpy_smx
            pretrained_cp_pred_sets, pretrained_cp_smooth_scores = self.pretrained_cp_predictor.predict(tt_scores)


            # realtime cp prediction
            tt_scores = realtime_tta_inference_munch.numpy_smx
            realtime_cp_pred_sets, realtime_cp_smooth_scores = self.realtime_cp_predictor.predict(tt_scores)
            

            # store include larger than
            self.pretrained_cp_include_larger_than = [self.pretrained_cp_predictor.get_include_larger_than()] * len(tt_dataloader.dataset)
            self.realtime_cp_include_larger_than = [self.realtime_cp_predictor.get_include_larger_than()] * len(tt_dataloader.dataset)



        if self.use_pseudo_coverage:

            # calculate pretrained cp weights
            if self.pretrained_cp_previous_weight is None:
                self.pretrained_cp_previous_weight = 0.5
                self.pretrained_cp_current_scalefactor = 1
                self.pretrained_cp_current_weight = self.pretrained_cp_previous_weight / self.pretrained_cp_current_scalefactor
            else:
                previous_coverage_gap = (1 - self.config.cp.alpha) - self.pretrained_cp_previous_coverage
                self.pretrained_cp_current_scalefactor = np.exp(previous_coverage_gap) * self.pretrained_cp_previous_scalefactor
                self.pretrained_cp_current_weight = self.pretrained_cp_previous_weight / self.pretrained_cp_current_scalefactor

            # calculate realtime cp weights
            if self.realtime_cp_previous_weight is None:
                self.realtime_cp_previous_weight = 0.5
                self.realtime_cp_current_scalefactor = 1
                self.realtime_cp_current_weight = self.realtime_cp_previous_weight / self.realtime_cp_current_scalefactor
            else:
                previous_coverage_gap = (1 - self.config.cp.alpha) - self.realtime_cp_previous_coverage
                self.realtime_cp_current_scalefactor = np.exp(previous_coverage_gap) * self.realtime_cp_previous_scalefactor
                self.realtime_cp_current_weight = self.realtime_cp_previous_weight / self.realtime_cp_current_scalefactor


            self.pretrained_cp_current_scalefactor = np.clip(self.pretrained_cp_current_scalefactor, 1e-50, 1e50)
            self.realtime_cp_current_scalefactor = np.clip(self.realtime_cp_current_scalefactor, 1e-50, 1e50)
            self.pretrained_cp_current_weight = np.clip(self.pretrained_cp_current_weight, 1e-50, 1e50)
            self.realtime_cp_current_weight = np.clip(self.realtime_cp_current_weight, 1e-50, 1e50)

            pretrained_weight_single_value = self.pretrained_cp_current_weight  
            realtime_weight_single_value = self.realtime_cp_current_weight

            pretrained_weights = np.array( [ pretrained_weight_single_value ] * len(cal_dataloader.dataset) )
            realtime_weights = np.array( [ realtime_weight_single_value ] * len(cal_dataloader.dataset) )

            # assign weights
            self.pretrained_cp_predictor.update_weights(pretrained_weights)
            self.realtime_cp_predictor.update_weights(realtime_weights)

            # store the cp calibration data weights
            self.pretrained_cp_cal_weights_mean = np.mean( self.pretrained_cp_predictor.weights )
            self.realtime_cp_cal_weights_mean = np.mean( self.realtime_cp_predictor.weights )

            # calibrate pretrained cp
            cal_scores = self.pretrained_cal_inference_munch.numpy_smx
            self.pretrained_cp_predictor.calibrate(cal_scores, self.cal_labels)
            
            # calibrate realtime cp
            cal_scores = realtime_cal_inference_munch.numpy_smx
            self.realtime_cp_predictor.calibrate(cal_scores, self.cal_labels)
            
            # pretrained cp prediction
            tt_scores = pretrained_tta_inference_munch.numpy_smx
            pretrained_cp_pred_sets, pretrained_cp_smooth_scores = self.pretrained_cp_predictor.predict(tt_scores)

            # realtime cp prediction
            tt_scores = realtime_tta_inference_munch.numpy_smx
            realtime_cp_pred_sets, realtime_cp_smooth_scores = self.realtime_cp_predictor.predict(tt_scores)
            
            # store include larger than
            self.pretrained_cp_include_larger_than = [self.pretrained_cp_predictor.get_include_larger_than()] * len(tt_dataloader.dataset)
            self.realtime_cp_include_larger_than = [self.realtime_cp_predictor.get_include_larger_than()] * len(tt_dataloader.dataset)

            # update pseudo coverage
            self.pretrained_cp_previous_coverage = self.pretrained_cp_predictor.evaluate(pretrained_cp_pred_sets, realtime_tta_inference_munch.numpy_pred_labels).cov
            self.realtime_cp_previous_coverage = self.realtime_cp_predictor.evaluate(realtime_cp_pred_sets, realtime_tta_inference_munch.numpy_pred_labels).cov
            self.pretrained_cp_pseudo_coverage = self.pretrained_cp_previous_coverage
            self.realtime_cp_pseudo_coverage = self.realtime_cp_previous_coverage

            # update scale factor
            self.pretrained_cp_previous_scalefactor = self.pretrained_cp_current_scalefactor
            self.realtime_cp_previous_scalefactor = self.realtime_cp_current_scalefactor
                    
            # update weights
            self.pretrained_cp_previous_weight = self.pretrained_cp_current_weight
            self.realtime_cp_previous_weight = self.realtime_cp_current_weight


        assert len(self.pretrained_cp_include_larger_than) == len(tt_dataloader.dataset)
        assert len(self.realtime_cp_include_larger_than) == len(tt_dataloader.dataset)   

        # store metrics
        self.pretrained_cp_pred_sets = pretrained_cp_pred_sets
        self.pretrained_cp_smooth_scores = pretrained_cp_smooth_scores
        self.pretrained_cp_eval_res = self.pretrained_cp_predictor.evaluate(pretrained_cp_pred_sets, tt_labels)

        self.realtime_cp_pred_sets = realtime_cp_pred_sets
        self.realtime_cp_smooth_scores = realtime_cp_smooth_scores
        self.realtime_cp_eval_res = self.realtime_cp_predictor.evaluate(realtime_cp_pred_sets, tt_labels)

        full_set_size = cal_scores.shape[1]



        # add a logic here for determine how many data to select
        cur_feats = pretrained_tta_inference_munch.numpy_feats

        if self.previous_feats is None: # this means that we just met the first batch

            # active selection
            oracle_num = self.oracle_num_new_domain
            pseudo_num = self.pseudo_num_per_batch

            # store the feats
            self.previous_feats = cur_feats
        else:

            # calculate dss
            dss = domain_shift_degree_eval.domain_shift_signal_using_feats(self.previous_feats, cur_feats)
            
            # calculate moving average
            self.dss_history.append(dss)
            moving_average = np.mean(self.dss_history).item()

            # detect a new domain
            if dss > self.dss_k * moving_average:
                oracle_num = self.oracle_num_new_domain
            else:
                oracle_num = self.oracle_num_per_batch
            pseudo_num = self.pseudo_num_per_batch

            # replace the previous batch feats
            self.previous_feats = cur_feats    

        if self.use_smooth:

            # get oracle indices
            smooth_scores = realtime_cp_smooth_scores
            sorted_smooth_scores = np.sort(smooth_scores)[:, ::-1]
            top_k_elements = sorted_smooth_scores[:, :self.smooth_topK_avg]
            avg_top_k = np.mean(top_k_elements, axis=1)
            _, oracle_indices = self.get_top_bottom_indices(arr=avg_top_k, topN=pseudo_num, bottomM=oracle_num)

            # get pseudo indices
            smooth_scores = pretrained_cp_smooth_scores
            sorted_smooth_scores = np.sort(smooth_scores)[:, ::-1]
            top_k_elements = sorted_smooth_scores[:, :self.smooth_topK_avg]
            avg_top_k = np.mean(top_k_elements, axis=1)
            pseudo_indices, _ = self.get_top_bottom_indices(arr=avg_top_k, topN=pseudo_num, bottomM=oracle_num)
            
        else:

            # get oracle indices
            pred_set_sizes = np.array( [len(d) if d else full_set_size + 1 for d in realtime_cp_pred_sets] )
            oracle_indices, _ = self.get_top_bottom_indices(arr=pred_set_sizes, topN=oracle_num, bottomM=pseudo_num)
            
            # get pseudo indices
            pred_set_sizes = np.array( [len(d) if d else full_set_size + 1 for d in pretrained_cp_pred_sets] )
            _, pseudo_indices = self.get_top_bottom_indices(arr=pred_set_sizes, topN=oracle_num, bottomM=pseudo_num)


        # ########################## use different selection method (comment out for random selection)  ##########################
        # print("Using Random Selection")
        # batch_size = realtime_cp_smooth_scores.shape[0]
        # random_indices_options = np.random.choice(batch_size, oracle_num + pseudo_num, replace=False)
        # oracle_indices = random_indices_options[:oracle_num]
        # pseudo_indices = random_indices_options[oracle_num:]


        self.pseudo_indices = pseudo_indices
        self.oracle_indices = oracle_indices
        
        # calculate before training loss per sample
        self.before_training_oracle_loss_per_sample = 0
        self.before_training_pseudo_loss_per_sample = 0
        self.before_training_oracle_loss_per_sample = self.get_before_training_loss_given_samples(torch_data[oracle_indices], torch_true_labels[oracle_indices])
        self.before_training_pseudo_loss_per_sample = self.get_before_training_loss_given_samples(torch_data[pseudo_indices], torch_true_labels[pseudo_indices])


        # update storage
        self.oracle_labeled_storage = self.update_storage(self.oracle_labeled_storage, 
                                                          torch_data[oracle_indices], 
                                                          torch_feats[oracle_indices],
                                                          torch_true_labels[oracle_indices], 
                                                          torch_pred_labels[oracle_indices])
        
        self.pseudo_labeled_storage = self.update_storage(self.pseudo_labeled_storage, 
                                                          torch_data[pseudo_indices], 
                                                          torch_feats[pseudo_indices],
                                                          torch_true_labels[pseudo_indices], 
                                                          torch_pred_labels[pseudo_indices])

        # calculate the number of correctly/incorrectly predicted data
        self.oracle_correct_num = np.sum(torch_pred_labels[oracle_indices].cpu().numpy() == torch_true_labels[oracle_indices].cpu().numpy())
        self.oracle_incorrect_num = np.sum(torch_pred_labels[oracle_indices].cpu().numpy() != torch_true_labels[oracle_indices].cpu().numpy())
        self.pseudo_correct_num = np.sum(torch_pred_labels[pseudo_indices].cpu().numpy() == torch_true_labels[pseudo_indices].cpu().numpy())
        self.pseudo_incorrect_num = np.sum(torch_pred_labels[pseudo_indices].cpu().numpy() != torch_true_labels[pseudo_indices].cpu().numpy())


    def train_model(self):
        if self.train_method == "simple":
            self.simple_train()
        elif self.train_method == "separate":
            self.separate_train()
        else:
            self.simatta_train()





################################ Using ExTHR CP predictor #########################################
@register.atta_alg_register
class CPATTA_ExTHR(cpatta_base.CPATTA_Base):

    def __init__(self, config):

        super().__init__(config)
        
        # we need two cp predictors
        self.realtime_cp_predictor = register.cp_predictors[config.cp.cp_name](config)
        self.pretrained_cp_predictor = register.cp_predictors[config.cp.cp_name](config)

        # we store two lr values for seperate training
        self.oracle_lr = config.train.oracle_lr
        self.pseudo_lr = config.train.pseudo_lr

        # handle how to use the cp result
        self.use_smooth = config.atta.use_smooth
        self.smooth_topK_avg = config.atta.smooth_topK_avg
        
        # record the config
        self.config = config

        # set up variables for dss, this is used for detecting domain change
        self.dss_k = config.atta.dss_k_value
        self.dss_history = []
        self.previous_feats = None

        # set up variables for selecting more data when encountering a new domain for the first time
        self.oracle_num_new_domain = config.atta.oracle_num_new_domain

        # we also need two sets of cp corresponding variables
        self.realtime_cp_eval_res = None
        self.realtime_cp_include_larger_than = None
        self.realtime_cp_pred_sets = None
        self.realtime_cp_smooth_scores = None

        self.pretrained_cp_eval_res = None
        self.pretrained_cp_include_larger_than = None
        self.pretrained_cp_pred_sets = None
        self.pretrained_cp_smooth_scores = None

        self.pretrained_cp_calibrated = False
        self.pretrained_cal_inference_munch = None
        self.cal_labels = None


    def select_data(self, tt_dataloader, cal_dataloader=None):

        assert cal_dataloader is not None

        # model inference
        if self.pretrained_cal_inference_munch is None:
            self.pretrained_cal_inference_munch = model_utils.model_inference(self.teacher, cal_dataloader, self.device)
        realtime_cal_inference_munch = model_utils.model_inference(self.model, cal_dataloader, self.device)
        pretrained_tta_inference_munch = model_utils.model_inference(self.teacher, tt_dataloader, self.device)
        realtime_tta_inference_munch = model_utils.model_inference(self.model, tt_dataloader, self.device)
        
        # store cal_labels
        if self.cal_labels is None:
            self.cal_labels = data_utils.extract_labels(cal_dataloader)
        
        tt_labels = data_utils.extract_labels(tt_dataloader)
        
        torch_true_labels = torch.cat([labels for _, labels in tt_dataloader], dim=0)
        torch_data = torch.cat([d for d, _ in tt_dataloader], dim=0)
        torch_pred_labels = realtime_tta_inference_munch.torch_pred_labels.cpu()
        torch_feats = realtime_tta_inference_munch.torch_feats.cpu()
        torch_pred_labels_pretrained = pretrained_tta_inference_munch.torch_pred_labels.cpu()

        # calibrate pretrained cp
        if not self.pretrained_cp_calibrated:
            cal_scores = self.pretrained_cal_inference_munch.numpy_smx
            self.pretrained_cp_predictor.calibrate(cal_scores, self.cal_labels)
            self.pretrained_cp_calibrated = True

        # calibrate realtime cp
        cal_scores = realtime_cal_inference_munch.numpy_smx
        self.realtime_cp_predictor.calibrate(cal_scores, self.cal_labels)
        
        # pretrained cp prediction
        tt_scores = pretrained_tta_inference_munch.numpy_smx
        pretrained_cp_pred_sets, pretrained_cp_smooth_scores = self.pretrained_cp_predictor.predict(tt_scores)

        # realtime cp prediction
        tt_scores = realtime_tta_inference_munch.numpy_smx
        realtime_cp_pred_sets, realtime_cp_smooth_scores = self.realtime_cp_predictor.predict(tt_scores)

        # record pretrained cp metrics
        self.pretrained_cp_eval_res = self.pretrained_cp_predictor.evaluate(pretrained_cp_pred_sets, tt_labels)
        self.pretrained_cp_include_larger_than = [self.pretrained_cp_predictor.get_include_larger_than()] * len(tt_dataloader.dataset)
        self.pretrained_cp_pred_sets = pretrained_cp_pred_sets
        self.pretrained_cp_smooth_scores = pretrained_cp_smooth_scores

        # record realtime cp metrics
        self.realtime_cp_eval_res = self.realtime_cp_predictor.evaluate(realtime_cp_pred_sets, tt_labels)
        self.realtime_cp_include_larger_than = [self.realtime_cp_predictor.get_include_larger_than()] * len(tt_dataloader.dataset)
        self.realtime_cp_pred_sets = realtime_cp_pred_sets
        self.realtime_cp_smooth_scores = realtime_cp_smooth_scores

        # add a logic here for determine how many data to select
        cur_feats = pretrained_tta_inference_munch.numpy_feats

        if self.previous_feats is None: # this means that we just met the first batch

            # active selection
            oracle_num = self.oracle_num_new_domain
            pseudo_num = self.pseudo_num_per_batch

            # store the feats
            self.previous_feats = cur_feats
        else:

            # calculate dss
            dss = domain_shift_degree_eval.domain_shift_signal_using_feats(self.previous_feats, cur_feats)
            
            # calculate moving average
            self.dss_history.append(dss)
            moving_average = np.mean(self.dss_history).item()

            # detect a new domain
            if dss > self.dss_k * moving_average:
                oracle_num = self.oracle_num_new_domain
            else:
                oracle_num = self.oracle_num_per_batch
            pseudo_num = self.pseudo_num_per_batch

            # replace the previous batch feats
            self.previous_feats = cur_feats    


        # determine how many data to select
        full_set_size = cal_scores.shape[1]
        if self.use_smooth:

            # get oracle indices
            smooth_scores = realtime_cp_smooth_scores
            sorted_smooth_scores = np.sort(smooth_scores)[:, ::-1]
            top_k_elements = sorted_smooth_scores[:, :self.smooth_topK_avg]
            avg_top_k = np.mean(top_k_elements, axis=1)
            _, oracle_indices = self.get_top_bottom_indices(arr=avg_top_k, topN=pseudo_num, bottomM=oracle_num)

            # get pseudo indices
            smooth_scores = pretrained_cp_smooth_scores
            sorted_smooth_scores = np.sort(smooth_scores)[:, ::-1]
            top_k_elements = sorted_smooth_scores[:, :self.smooth_topK_avg]
            avg_top_k = np.mean(top_k_elements, axis=1)
            pseudo_indices, _ = self.get_top_bottom_indices(arr=avg_top_k, topN=pseudo_num, bottomM=oracle_num)
            
        else:

            # get oracle indices
            pred_set_sizes = np.array( [len(d) if d else full_set_size + 1 for d in realtime_cp_pred_sets] )
            oracle_indices, _ = self.get_top_bottom_indices(arr=pred_set_sizes, topN=oracle_num, bottomM=pseudo_num)
            
            # get pseudo indices
            pred_set_sizes = np.array( [len(d) if d else full_set_size + 1 for d in pretrained_cp_pred_sets] )
            _, pseudo_indices = self.get_top_bottom_indices(arr=pred_set_sizes, topN=oracle_num, bottomM=pseudo_num)    

  
        self.pseudo_indices = pseudo_indices
        self.oracle_indices = oracle_indices
        
        # update storage
        self.oracle_labeled_storage = self.update_storage(self.oracle_labeled_storage, 
                                                          torch_data[oracle_indices], 
                                                          torch_feats[oracle_indices],
                                                          torch_true_labels[oracle_indices], 
                                                          torch_pred_labels[oracle_indices])

        self.pseudo_labeled_storage = self.update_storage(self.pseudo_labeled_storage, 
                                                          torch_data[pseudo_indices], 
                                                          torch_feats[pseudo_indices],
                                                          torch_true_labels[pseudo_indices], 
                                                          torch_pred_labels_pretrained[pseudo_indices])

        # calculate the number of correctly/incorrectly predicted data
        self.oracle_correct_num = np.sum(torch_pred_labels[oracle_indices].cpu().numpy() == torch_true_labels[oracle_indices].cpu().numpy())
        self.oracle_incorrect_num = np.sum(torch_pred_labels[oracle_indices].cpu().numpy() != torch_true_labels[oracle_indices].cpu().numpy())
        self.pseudo_correct_num = np.sum(torch_pred_labels[pseudo_indices].cpu().numpy() == torch_true_labels[pseudo_indices].cpu().numpy())
        self.pseudo_incorrect_num = np.sum(torch_pred_labels[pseudo_indices].cpu().numpy() != torch_true_labels[pseudo_indices].cpu().numpy())
        

    def train_model(self):

        if self.train_method == "simple":
            self.simple_train()
        elif self.train_method == "separate":
            self.separate_train()
        else:
            self.simatta_train()





################################ Using QTC CP predictor #########################################


@register.atta_alg_register
class CPATTA_QTC(cpatta_base.CPATTA_Base):

    def __init__(self, config):
        super().__init__(config)
        
        # we need two cp predictors
        self.realtime_cp_predictor = register.cp_predictors[config.cp.cp_name](config)
        self.pretrained_cp_predictor = register.cp_predictors[config.cp.cp_name](config)

        # we store two ir values for seperate training
        self.oracle_lr = config.train.oracle_lr
        self.pseudo_lr = config.train.pseudo_lr

        # handle how to use the cp result
        self.use_smooth = config.atta.use_smooth
        self.smooth_topK_avg = config.atta.smooth_topK_avg
        
        # record the config
        self.config = config

        # set up variables for dss, this is used for detecting domain change
        self.dss_k = config.atta.dss_k_value
        self.dss_history = []
        self.previous_feats = None

        # set up variables for selecting more data when encountering a new domain for the first time
        self.oracle_num_new_domain = config.atta.oracle_num_new_domain

        # we also need two sets of cp corresponding variables
        self.realtime_cp_eval_res = None
        self.realtime_cp_include_larger_than = None
        self.realtime_cp_pred_sets = None
        self.realtime_cp_smooth_scores = None

        self.pretrained_cp_eval_res = None
        self.pretrained_cp_include_larger_than = None
        self.pretrained_cp_pred_sets = None
        self.pretrained_cp_smooth_scores = None

        self.pretrained_cp_calibrated = False
        self.pretrained_cal_inference_munch = None
        self.cal_labels = None



    def select_data(self, tt_dataloader, cal_dataloader=None):
        
        assert cal_dataloader is not None

        # model inference
        if self.pretrained_cal_inference_munch is None:
            self.pretrained_cal_inference_munch = model_utils.model_inference(self.teacher, cal_dataloader, self.device)
        realtime_cal_inference_munch = model_utils.model_inference(self.model, cal_dataloader, self.device)
        pretrained_tta_inference_munch = model_utils.model_inference(self.teacher, tt_dataloader, self.device)
        realtime_tta_inference_munch = model_utils.model_inference(self.model, tt_dataloader, self.device)
        
        # store cal_labels
        if self.cal_labels is None:
            self.cal_labels = data_utils.extract_labels(cal_dataloader)
        
        tt_labels = data_utils.extract_labels(tt_dataloader)
        
        torch_true_labels = torch.cat([labels for _, labels in tt_dataloader], dim=0)
        torch_data = torch.cat([d for d, _ in tt_dataloader], dim=0)
        torch_pred_labels = realtime_tta_inference_munch.torch_pred_labels.cpu()
        torch_feats = realtime_tta_inference_munch.torch_feats.cpu()
        torch_pred_labels_pretrained = pretrained_tta_inference_munch.torch_pred_labels.cpu()

        # calibrate pretrained cp
        self.pretrained_cp_predictor.calibrate(self.pretrained_cal_inference_munch.numpy_smx, self.cal_labels, pretrained_tta_inference_munch.numpy_smx)
        
        # pretrained cp prediction
        self.pretrained_cp_pred_sets, self.pretrained_cp_smooth_scores = self.pretrained_cp_predictor.predict(pretrained_tta_inference_munch.numpy_smx)
        
        # evaluate pretrained cp
        self.pretrained_cp_eval_res = self.pretrained_cp_predictor.evaluate(self.pretrained_cp_pred_sets, tt_labels)
        
        # pretrained cp include larger than
        self.pretrained_cp_include_larger_than = [self.pretrained_cp_predictor.get_include_larger_than()] * len(tt_dataloader.dataset)

        # calibrate realtime cp 
        self.realtime_cp_predictor.calibrate(realtime_cal_inference_munch.numpy_smx, self.cal_labels, realtime_tta_inference_munch.numpy_smx)

        # realtime cp prediction
        self.realtime_cp_pred_sets, self.realtime_cp_smooth_scores = self.realtime_cp_predictor.predict(realtime_tta_inference_munch.numpy_smx)

        # evaluate realtime cp
        self.realtime_cp_eval_res = self.realtime_cp_predictor.evaluate(self.realtime_cp_pred_sets, tt_labels)

        # realtime cp include larger than
        self.realtime_cp_include_larger_than = [self.realtime_cp_predictor.get_include_larger_than()] * len(tt_dataloader.dataset)


        # add a logic here for determine how many data to select
        cur_feats = pretrained_tta_inference_munch.numpy_feats

        if self.previous_feats is None: # this means that we just met the first batch

            # active selection
            oracle_num = self.oracle_num_new_domain
            pseudo_num = self.pseudo_num_per_batch

            # store the feats
            self.previous_feats = cur_feats
        else:

            # calculate dss
            dss = domain_shift_degree_eval.domain_shift_signal_using_feats(self.previous_feats, cur_feats)
            
            # calculate moving average
            self.dss_history.append(dss)
            moving_average = np.mean(self.dss_history).item()

            # detect a new domain
            if dss > self.dss_k * moving_average:
                oracle_num = self.oracle_num_new_domain
            else:
                oracle_num = self.oracle_num_per_batch
            pseudo_num = self.pseudo_num_per_batch

            # replace the previous batch feats
            self.previous_feats = cur_feats    


        # determine how many data to select
        realtime_cp_smooth_scores = self.realtime_cp_smooth_scores
        pretrained_cp_smooth_scores = self.pretrained_cp_smooth_scores
        realtime_cp_pred_sets = self.realtime_cp_pred_sets
        pretrained_cp_pred_sets = self.pretrained_cp_pred_sets
        cal_scores = self.pretrained_cal_inference_munch.numpy_smx

        full_set_size = cal_scores.shape[1]
        if self.use_smooth:

            # get oracle indices
            smooth_scores = realtime_cp_smooth_scores
            sorted_smooth_scores = np.sort(smooth_scores)[:, ::-1]
            top_k_elements = sorted_smooth_scores[:, :self.smooth_topK_avg]
            avg_top_k = np.mean(top_k_elements, axis=1)
            _, oracle_indices = self.get_top_bottom_indices(arr=avg_top_k, topN=pseudo_num, bottomM=oracle_num)

            # get pseudo indices
            smooth_scores = pretrained_cp_smooth_scores
            sorted_smooth_scores = np.sort(smooth_scores)[:, ::-1]
            top_k_elements = sorted_smooth_scores[:, :self.smooth_topK_avg]
            avg_top_k = np.mean(top_k_elements, axis=1)
            pseudo_indices, _ = self.get_top_bottom_indices(arr=avg_top_k, topN=pseudo_num, bottomM=oracle_num)
            
        else:

            # get oracle indices
            pred_set_sizes = np.array( [len(d) if d else full_set_size + 1 for d in realtime_cp_pred_sets] )
            oracle_indices, _ = self.get_top_bottom_indices(arr=pred_set_sizes, topN=oracle_num, bottomM=pseudo_num)
            
            # get pseudo indices
            pred_set_sizes = np.array( [len(d) if d else full_set_size + 1 for d in pretrained_cp_pred_sets] )
            _, pseudo_indices = self.get_top_bottom_indices(arr=pred_set_sizes, topN=oracle_num, bottomM=pseudo_num)    

  
        self.pseudo_indices = pseudo_indices
        self.oracle_indices = oracle_indices
        
        # update storage
        self.oracle_labeled_storage = self.update_storage(self.oracle_labeled_storage, 
                                                          torch_data[oracle_indices], 
                                                          torch_feats[oracle_indices],
                                                          torch_true_labels[oracle_indices], 
                                                          torch_pred_labels[oracle_indices])
        
        self.pseudo_labeled_storage = self.update_storage(self.pseudo_labeled_storage, 
                                                          torch_data[pseudo_indices], 
                                                          torch_feats[pseudo_indices],
                                                          torch_true_labels[pseudo_indices], 
                                                          torch_pred_labels[pseudo_indices])

        # calculate the number of correctly/incorrectly predicted data
        self.oracle_correct_num = np.sum(torch_pred_labels[oracle_indices].cpu().numpy() == torch_true_labels[oracle_indices].cpu().numpy())
        self.oracle_incorrect_num = np.sum(torch_pred_labels[oracle_indices].cpu().numpy() != torch_true_labels[oracle_indices].cpu().numpy())
        self.pseudo_correct_num = np.sum(torch_pred_labels[pseudo_indices].cpu().numpy() == torch_true_labels[pseudo_indices].cpu().numpy())
        self.pseudo_incorrect_num = np.sum(torch_pred_labels[pseudo_indices].cpu().numpy() != torch_true_labels[pseudo_indices].cpu().numpy())


    def train_model(self):
        if self.train_method == "simple":
            self.simple_train()
        elif self.train_method == "separate":
            self.separate_train()
        else:
            self.simatta_train()