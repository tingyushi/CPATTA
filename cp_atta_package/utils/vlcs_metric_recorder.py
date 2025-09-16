"""
This code file contains functions to record metrics while adapting different domains using different conformal predictors
"""
import pandas as pd
import os
import numpy as np
import time
import sys

from cp_atta_package.utils import process_ymal, model_utils, data_utils
from cp_atta_package.utils.register import register
from cp_atta_package.model import vlcs_model_loader
from cp_atta_package.data import vlcs_data_loader
from cp_atta_package.atta_algs import cpatta
from cp_atta_package.cp import nexcrc, qtc, exthr




def nexcrc_metric_recorder(config, data_streams, domainwise_dataloaders, 
                                    cal_dataloader, sname, folder_path, 
                                    data_stream_names, domain_names, 
                                    save_csv=True):

    # source domain name
    source_domain_name = domain_names[0]

    # store domainwise datastream results
    row_items = ['realtime_acc', 'budgets'] + domain_names
    col_items = domain_names
    continue_result_df = pd.DataFrame(index=row_items, columns=col_items, dtype=float)

    # store random datastream results
    row_items = ['realtime_acc', 'budgets'] + domain_names
    col_items = ['pre'] + [f"split_{i}" for i in range(4)] # modify here for different number of splits
    random_result_df = pd.DataFrame(index=row_items, columns=col_items, dtype=float)   

    # batch level columns
    batch_level_columns = ["batch_index", 
                           "batch_size", 
                           "domain_name", 
                           "realtime_model_acc", 

                           "pretrained_cp_cov", 
                           "pretrained_cp_pseudo_cov",
                           "pretrained_cp_pred_set_sizes_avg", 
                           "pretrained_cp_num_empty_sets", 
                           "pretrained_cp_include_larger_than_avg", 
                           "pretrained_cp_weights_mean",
                           "pretrained_cp_scalefactor",
                           "pretrained_cal_entropy",
                           "pretrained_tta_entropy",

                           "realtime_cp_cov", 
                           "realtime_cp_pseudo_cov",
                           "realtime_cp_pred_set_sizes_avg",
                           "realtime_cp_num_empty_sets", 
                           "realtime_cp_include_larger_than_avg",
                           "realtime_cp_weights_mean", 
                           "realtime_cp_scalefactor",
                           "realtime_cal_entropy",
                           "realtime_tta_entropy",

                           "oracle_lr",
                           "pseudo_lr",
                           "lr",
                           
                           "oracle_loss_coef",
                           "pseudo_loss_coef",
                           
                           "before_training_oracle_loss_sum",
                           "before_training_pseudo_loss_sum",
                           "during_training_oracle_loss_avg",
                           "during_training_pseudo_loss_avg",
                           
                           "num_oracle_labeled", 
                           "num_pseudo_labeled"]


    # sample level columns
    sample_level_columns = ["batch_id", 
                            "domain_name",
                            "sample_index", 
                            "labeling_method",
                            
                            "realtime_model_pred_label",
                            "true_label", 
                            
                            "pretrained_cp_include_larger_than",
                            "pretrained_cp_pred_set", 
                            "pretrained_cp_smooth_score", 
                            
                            "realtime_cp_include_larger_than",
                            "realtime_cp_pred_set", 
                            "realtime_cp_smooth_score",]


    for data_stream_name in data_stream_names:

        print(f"\n========== Working on {data_stream_name} ==========")

        atta_obj = register.atta_algs[config.atta.alg_name](config)
        
        print(f"========== atta alg name: {type(atta_obj).__name__} ==========")
        print(f"========== pretrained cp name: {type(atta_obj.pretrained_cp_predictor).__name__} ==========")
        print(f"========== realtime cp name: {type(atta_obj.pretrained_cp_predictor).__name__} ==========")

        # record pre-adaptation information
        if data_stream_name == "domainwise":
            for ele in domainwise_dataloaders:
                inference_res = model_utils.model_inference( atta_obj.model, ele.dataloader, config.device )
                continue_result_df.loc[ele.domain_name, source_domain_name] = inference_res.acc
            print(f'\n{continue_result_df.round(4).to_markdown()}\n')
        else:
            for ele in domainwise_dataloaders:
                inference_res = model_utils.model_inference( atta_obj.model, ele.dataloader, config.device )
                random_result_df.loc[ele.domain_name, "pre"] = inference_res.acc
            print(f'\n{random_result_df.round(4).to_markdown()}\n')


        batch_level_list = []
        sample_level_list = []
        
        # used for calculating accuracy whe changing domains
        correct = 0
        num_samples = 0

        for idx, batch in enumerate(data_streams[data_stream_name]):


            ########## Store batch level information ##########

            true_labels = data_utils.extract_labels(batch.dataloader)
            realtime_cur_munch = model_utils.model_inference( atta_obj.model, batch.dataloader, config.device )
            batch_size = len(batch.dataloader.dataset)


            batch_level_new_dict = {d:"NA" for d in batch_level_columns}
            batch_level_new_dict['batch_index'] = idx
            batch_level_new_dict['batch_size'] = batch_size
            batch_level_new_dict['domain_name'] = batch.domain_name
            batch_level_new_dict['realtime_model_acc'] = realtime_cur_munch.acc
            batch_level_new_dict['num_oracle_labeled'] = 0
            batch_level_new_dict['num_pseudo_labeled'] = 0
            
            if batch.domain_name != source_domain_name:
                
                correct += (realtime_cur_munch.acc * len(batch.dataloader.dataset))
                num_samples += len(batch.dataloader.dataset)

                # select data
                atta_obj.select_data(batch.dataloader, cal_dataloader)

                # get before training loss
                before_training_oracle_loss, before_training_pseudo_loss = atta_obj.get_before_training_loss()

                # get batch level metrics
                batch_level_new_dict = {d:"NA" for d in batch_level_columns}
                batch_level_new_dict['batch_index'] = idx
                batch_level_new_dict['batch_size'] = batch_size
                batch_level_new_dict['domain_name'] = batch.domain_name
                batch_level_new_dict['realtime_model_acc'] = realtime_cur_munch.acc

                
                batch_level_new_dict['pretrained_cp_cov'] = atta_obj.pretrained_cp_eval_res.cov
                batch_level_new_dict['pretrained_cp_pseudo_cov'] = getattr(atta_obj, "pretrained_cp_pseudo_coverage", None)
                batch_level_new_dict['pretrained_cp_pred_set_sizes_avg'] = atta_obj.pretrained_cp_eval_res.pred_set_size
                batch_level_new_dict['pretrained_cp_num_empty_sets'] = atta_obj.pretrained_cp_eval_res.num_empty_sets
                batch_level_new_dict['pretrained_cp_include_larger_than_avg'] = np.mean(atta_obj.pretrained_cp_include_larger_than).item()
                batch_level_new_dict['pretrained_cp_weights_mean'] = atta_obj.pretrained_cp_cal_weights_mean
                batch_level_new_dict['pretrained_cp_scalefactor'] = getattr(atta_obj, "pretrained_cp_current_scalefactor", None)
                batch_level_new_dict['pretrained_cal_entropy'] = getattr(atta_obj, "pretrained_cal_entropy", None)
                batch_level_new_dict['pretrained_tta_entropy'] = getattr(atta_obj, "pretrained_tta_entropy", None)


                batch_level_new_dict['realtime_cp_cov'] = atta_obj.realtime_cp_eval_res.cov
                batch_level_new_dict['realtime_cp_pseudo_cov'] = getattr(atta_obj, "realtime_cp_pseudo_coverage", None)
                batch_level_new_dict['realtime_cp_pred_set_sizes_avg'] = atta_obj.realtime_cp_eval_res.pred_set_size
                batch_level_new_dict['realtime_cp_num_empty_sets'] = atta_obj.realtime_cp_eval_res.num_empty_sets
                batch_level_new_dict['realtime_cp_include_larger_than_avg'] = np.mean(atta_obj.realtime_cp_include_larger_than).item()
                batch_level_new_dict['realtime_cp_weights_mean'] = atta_obj.realtime_cp_cal_weights_mean
                batch_level_new_dict['realtime_cp_scalefactor'] = getattr( atta_obj, "realtime_cp_current_scalefactor", None)
                batch_level_new_dict['realtime_cal_entropy'] = getattr(atta_obj, "realtime_cal_entropy", None)
                batch_level_new_dict['realtime_tta_entropy'] = getattr(atta_obj, "realtime_tta_entropy", None)


                oracle_indices = atta_obj.oracle_indices.tolist()
                pseudo_indices = atta_obj.pseudo_indices.tolist()
                temp = oracle_indices + pseudo_indices
                no_indices = [i for i in range(batch_size) if i not in temp]

                ##### Store oracle labeled data #####
                for sample_idx in oracle_indices:
                    sample_level_list.append([idx, 
                                              batch.domain_name, 
                                              sample_idx, 
                                              "oracle", 
                                              realtime_cur_munch.numpy_pred_labels[sample_idx].item(),  
                                              true_labels[sample_idx].item(), 
                                              atta_obj.pretrained_cp_include_larger_than[sample_idx],
                                              atta_obj.pretrained_cp_pred_sets[sample_idx], 
                                              atta_obj.pretrained_cp_smooth_scores[sample_idx].tolist(), 
                                              atta_obj.realtime_cp_include_larger_than[sample_idx],
                                              atta_obj.realtime_cp_pred_sets[sample_idx], 
                                              atta_obj.realtime_cp_smooth_scores[sample_idx].tolist()])
                    

                ##### Store pseudo labeled data #####
                for sample_idx in pseudo_indices:
                    sample_level_list.append([idx, 
                                              batch.domain_name, 
                                              sample_idx, 
                                              "pseudo", 
                                              realtime_cur_munch.numpy_pred_labels[sample_idx].item(),  
                                              true_labels[sample_idx].item(), 
                                              atta_obj.pretrained_cp_include_larger_than[sample_idx],
                                              atta_obj.pretrained_cp_pred_sets[sample_idx], 
                                              atta_obj.pretrained_cp_smooth_scores[sample_idx].tolist(), 
                                              atta_obj.realtime_cp_include_larger_than[sample_idx],
                                              atta_obj.realtime_cp_pred_sets[sample_idx], 
                                              atta_obj.realtime_cp_smooth_scores[sample_idx].tolist()])


                ##### Store no labeled data #####
                for sample_idx in no_indices:
                    sample_level_list.append([idx, 
                                              batch.domain_name, 
                                              sample_idx, 
                                              "no", 
                                              realtime_cur_munch.numpy_pred_labels[sample_idx].item(),  
                                              true_labels[sample_idx].item(), 
                                              atta_obj.pretrained_cp_include_larger_than[sample_idx],
                                              atta_obj.pretrained_cp_pred_sets[sample_idx], 
                                              atta_obj.pretrained_cp_smooth_scores[sample_idx].tolist(), 
                                              atta_obj.realtime_cp_include_larger_than[sample_idx],
                                              atta_obj.realtime_cp_pred_sets[sample_idx], 
                                              atta_obj.realtime_cp_smooth_scores[sample_idx].tolist()])


                atta_obj.train_model()
                
                # append pseudo/oracle lr
                batch_level_new_dict['lr'] = atta_obj.lr
                batch_level_new_dict['oracle_lr'] = atta_obj.oracle_lr
                batch_level_new_dict['pseudo_lr'] = atta_obj.pseudo_lr
            
                # append 4 types of loss
                batch_level_new_dict['before_training_oracle_loss_sum'] = before_training_oracle_loss
                batch_level_new_dict['before_training_pseudo_loss_sum'] = before_training_pseudo_loss
                batch_level_new_dict['during_training_oracle_loss_avg'] = atta_obj.during_training_oracle_loss
                batch_level_new_dict['during_training_pseudo_loss_avg'] = atta_obj.during_training_pseudo_loss
                
                # append number of oracle data and pseudo data
                batch_level_new_dict['num_oracle_labeled'] = atta_obj.oracle_labeled_storage.num_elem()
                batch_level_new_dict['num_pseudo_labeled'] = atta_obj.pseudo_labeled_storage.num_elem()
                
                if atta_obj.pseudo_loss_coef is not None: batch_level_new_dict['pseudo_loss_coef'] = atta_obj.pseudo_loss_coef
                if atta_obj.oracle_loss_coef is not None: batch_level_new_dict['oracle_loss_coef'] = atta_obj.oracle_loss_coef


            batch_level_new_list = [batch_level_new_dict[d] for d in batch_level_columns]
            batch_level_list.append( batch_level_new_list )
            
            ########## print data ##########
            common_format = (
                "Batch Index: {:<4} -- "
                "Domain Name: {:<12} -- "
                "num_oracle_labeled: {:<8} -- "
                "num_pseudo_labeled: {:<8} -- "
                "pretrained_ILT_avg: {:<10} --"
                "realtime_ILT_avg: {:<10} --"
                "Acc: {:<4.3f} -- "
                "pretrained_weights_mean: {:<10.6f} -- "
                "realtime_weights_mean: {:<10.6f}")

            # Adjust float formatting only if needed
            if batch.domain_name != source_domain_name:
                pretrained_fmt = "{:<10.6f}"
                realtime_fmt = "{:<10.6f}"
            else:
                pretrained_fmt = "{:<10}"
                realtime_fmt = "{:<10}"

            # Final format string
            s = common_format.format(
                idx,
                batch.domain_name,
                batch_level_new_dict['num_oracle_labeled'],
                batch_level_new_dict['num_pseudo_labeled'],
                pretrained_fmt.format(batch_level_new_dict['pretrained_cp_include_larger_than_avg']),
                realtime_fmt.format(batch_level_new_dict['realtime_cp_include_larger_than_avg']),
                realtime_cur_munch.acc,
                atta_obj.pretrained_cp_cal_weights_mean,
                atta_obj.realtime_cp_cal_weights_mean)

            print(s)

     
            # store pre-post information
            if idx != ( len(data_streams[data_stream_name]) - 1 ): # test if not the last batch
                next_domain_name = data_streams[data_stream_name][idx + 1].domain_name
                test_domain_name = batch.domain_name
                if (next_domain_name != test_domain_name) and (test_domain_name != source_domain_name): 
                    if data_stream_name == "domainwise":
                        for ele in domainwise_dataloaders:
                            inference_res = model_utils.model_inference( atta_obj.model, ele.dataloader, config.device )
                            continue_result_df.loc[ele.domain_name, test_domain_name] = inference_res.acc
                        continue_result_df.loc["realtime_acc", test_domain_name] = correct / num_samples
                        continue_result_df.loc["budgets", test_domain_name] = batch_level_new_list[-2]
                        print(f'\n{continue_result_df.round(4).to_markdown()}\n')
                    else:
                        for ele in domainwise_dataloaders:
                            inference_res = model_utils.model_inference( atta_obj.model, ele.dataloader, config.device )
                            random_result_df.loc[ele.domain_name, test_domain_name] = inference_res.acc
                        random_result_df.loc["realtime_acc", test_domain_name] = correct / num_samples
                        random_result_df.loc["budgets", test_domain_name] = batch_level_new_list[-2]
                        print(f'\n{random_result_df.round(4).to_markdown()}\n')
                    
                    correct = 0
                    num_samples = 0
            else: # test if the last batch
                test_domain_name = batch.domain_name
                if data_stream_name == "domainwise":
                        for ele in domainwise_dataloaders:
                            inference_res = model_utils.model_inference( atta_obj.model, ele.dataloader, config.device )
                            continue_result_df.loc[ele.domain_name, test_domain_name] = inference_res.acc
                        continue_result_df.loc["realtime_acc", test_domain_name] = correct / num_samples
                        continue_result_df.loc["budgets", test_domain_name] = batch_level_new_list[-2]
                        print(f'\n{continue_result_df.round(4).to_markdown()}\n')
                else:
                    for ele in domainwise_dataloaders:
                        inference_res = model_utils.model_inference( atta_obj.model, ele.dataloader, config.device )
                        random_result_df.loc[ele.domain_name, test_domain_name] = inference_res.acc
                    random_result_df.loc["realtime_acc", test_domain_name] = correct / num_samples
                    random_result_df.loc["budgets", test_domain_name] = batch_level_new_list[-2]
                    print(f'\n{random_result_df.round(4).to_markdown()}\n')
                
                correct = 0
                num_samples = 0


        # save dfs
        if save_csv:
            sname_ = data_stream_name + "_" + sname
            batch_level_df = pd.DataFrame(batch_level_list, columns=batch_level_columns)
            batch_level_df.to_csv(os.path.join( folder_path, sname_ + "_batch_level.csv"), index=False)
            sample_level_df = pd.DataFrame(sample_level_list, columns=sample_level_columns)
            sample_level_df.to_csv(os.path.join( folder_path, sname_ + "_sample_level.csv"), index=False)
            if data_stream_name == "domainwise":
                continue_result_df.to_csv(os.path.join( folder_path, sname_ + "_pre_post.csv"))
            else:
                random_result_df.to_csv(os.path.join( folder_path, sname_ + "_pre_post.csv"))

        print()

        atta_obj.destroy_model()
        del atta_obj



def exthr_metric_recorder(config, data_streams, domainwise_dataloaders, 
                          cal_dataloader, sname, folder_path, 
                          data_stream_names, domain_names, save_csv=True):

    # source domain name
    source_domain_name = domain_names[0]

    # store domainwise datastream results
    row_items = ['realtime_acc', 'budgets'] + domain_names
    col_items = domain_names
    continue_result_df = pd.DataFrame(index=row_items, columns=col_items, dtype=float)

    # store random datastream results
    row_items = ['realtime_acc', 'budgets'] + domain_names
    col_items = ['pre'] + [f"split_{i}" for i in range(4)] # modify here for different number of splits
    random_result_df = pd.DataFrame(index=row_items, columns=col_items, dtype=float)   

    # batch level columns
    batch_level_columns = ["batch_index", 
                           "batch_size", 
                           "domain_name", 
                           "realtime_model_acc", 

                           "pretrained_cp_cov", 
                           "pretrained_cp_pred_set_sizes_avg", 
                           "pretrained_cp_num_empty_sets", 
                           "pretrained_cp_include_larger_than_avg", 
                           
                           "realtime_cp_cov", 
                           "realtime_cp_pred_set_sizes_avg",
                           "realtime_cp_num_empty_sets", 
                           "realtime_cp_include_larger_than_avg",
                           
                           "oracle_lr",
                           "pseudo_lr",
                           "lr",
                           
                           "oracle_loss_coef",
                           "pseudo_loss_coef",
                           
                           "before_training_oracle_loss_sum",
                           "before_training_pseudo_loss_sum",
                           "during_training_oracle_loss_avg",
                           "during_training_pseudo_loss_avg",
                           
                           "num_oracle_labeled", 
                           "num_pseudo_labeled"]


    # sample level columns
    sample_level_columns = ["batch_id", 
                            "domain_name",
                            "sample_index", 
                            "labeling_method",
                            
                            "realtime_model_pred_label",
                            "true_label", 
                            
                            "pretrained_cp_include_larger_than",
                            "pretrained_cp_pred_set", 
                            "pretrained_cp_smooth_score", 
                            
                            "realtime_cp_include_larger_than",
                            "realtime_cp_pred_set", 
                            "realtime_cp_smooth_score",]


    for data_stream_name in data_stream_names:

        print(f"\n========== Working on {data_stream_name} ==========")

        atta_obj = register.atta_algs[config.atta.alg_name](config)
        
        print(f"========== atta alg name: {type(atta_obj).__name__} ==========")
        print(f"========== pretrained cp name: {type(atta_obj.pretrained_cp_predictor).__name__} ==========")
        print(f"========== realtime cp name: {type(atta_obj.pretrained_cp_predictor).__name__} ==========")

        # record pre-adaptation information
        if data_stream_name == "domainwise":
            for ele in domainwise_dataloaders:
                inference_res = model_utils.model_inference( atta_obj.model, ele.dataloader, config.device )
                continue_result_df.loc[ele.domain_name, source_domain_name] = inference_res.acc
            print(f'\n{continue_result_df.round(4).to_markdown()}\n')
        else:
            for ele in domainwise_dataloaders:
                inference_res = model_utils.model_inference( atta_obj.model, ele.dataloader, config.device )
                random_result_df.loc[ele.domain_name, "pre"] = inference_res.acc
            print(f'\n{random_result_df.round(4).to_markdown()}\n')


        batch_level_list = []
        sample_level_list = []
        
        # used for calculating accuracy whe changing domains
        correct = 0
        num_samples = 0

        for idx, batch in enumerate(data_streams[data_stream_name]):


            ########## Store batch level information ##########

            true_labels = data_utils.extract_labels(batch.dataloader)
            realtime_cur_munch = model_utils.model_inference( atta_obj.model, batch.dataloader, config.device )
            batch_size = len(batch.dataloader.dataset)


            batch_level_new_dict = {d:"NA" for d in batch_level_columns}
            batch_level_new_dict['batch_index'] = idx
            batch_level_new_dict['batch_size'] = batch_size
            batch_level_new_dict['domain_name'] = batch.domain_name
            batch_level_new_dict['realtime_model_acc'] = realtime_cur_munch.acc
            batch_level_new_dict['num_oracle_labeled'] = 0
            batch_level_new_dict['num_pseudo_labeled'] = 0
            
            if batch.domain_name != source_domain_name:
                
                correct += (realtime_cur_munch.acc * len(batch.dataloader.dataset))
                num_samples += len(batch.dataloader.dataset)

                # select data
                atta_obj.select_data(batch.dataloader, cal_dataloader)

                # get before training loss
                before_training_oracle_loss, before_training_pseudo_loss = atta_obj.get_before_training_loss()

                # get batch level metrics
                batch_level_new_dict = {d:"NA" for d in batch_level_columns}
                batch_level_new_dict['batch_index'] = idx
                batch_level_new_dict['batch_size'] = batch_size
                batch_level_new_dict['domain_name'] = batch.domain_name
                batch_level_new_dict['realtime_model_acc'] = realtime_cur_munch.acc
                
                batch_level_new_dict['pretrained_cp_cov'] = atta_obj.pretrained_cp_eval_res.cov
                batch_level_new_dict['pretrained_cp_pred_set_sizes_avg'] = atta_obj.pretrained_cp_eval_res.pred_set_size
                batch_level_new_dict['pretrained_cp_num_empty_sets'] = atta_obj.pretrained_cp_eval_res.num_empty_sets
                batch_level_new_dict['pretrained_cp_include_larger_than_avg'] = np.mean(atta_obj.pretrained_cp_include_larger_than).item()

                batch_level_new_dict['realtime_cp_cov'] = atta_obj.realtime_cp_eval_res.cov
                batch_level_new_dict['realtime_cp_pred_set_sizes_avg'] = atta_obj.realtime_cp_eval_res.pred_set_size
                batch_level_new_dict['realtime_cp_num_empty_sets'] = atta_obj.realtime_cp_eval_res.num_empty_sets
                batch_level_new_dict['realtime_cp_include_larger_than_avg'] = np.mean(atta_obj.realtime_cp_include_larger_than).item()


                oracle_indices = atta_obj.oracle_indices.tolist()
                pseudo_indices = atta_obj.pseudo_indices.tolist()
                temp = oracle_indices + pseudo_indices
                no_indices = [i for i in range(batch_size) if i not in temp]

                ##### Store oracle labeled data #####
                for sample_idx in oracle_indices:
                    sample_level_list.append([idx, 
                                              batch.domain_name, 
                                              sample_idx, 
                                              "oracle", 
                                              realtime_cur_munch.numpy_pred_labels[sample_idx].item(),  
                                              true_labels[sample_idx].item(), 
                                              atta_obj.pretrained_cp_include_larger_than[sample_idx],
                                              atta_obj.pretrained_cp_pred_sets[sample_idx], 
                                              atta_obj.pretrained_cp_smooth_scores[sample_idx].tolist(), 
                                              atta_obj.realtime_cp_include_larger_than[sample_idx],
                                              atta_obj.realtime_cp_pred_sets[sample_idx], 
                                              atta_obj.realtime_cp_smooth_scores[sample_idx].tolist()])
                    

                ##### Store pseudo labeled data #####
                for sample_idx in pseudo_indices:
                    sample_level_list.append([idx, 
                                              batch.domain_name, 
                                              sample_idx, 
                                              "pseudo", 
                                              realtime_cur_munch.numpy_pred_labels[sample_idx].item(),  
                                              true_labels[sample_idx].item(), 
                                              atta_obj.pretrained_cp_include_larger_than[sample_idx],
                                              atta_obj.pretrained_cp_pred_sets[sample_idx], 
                                              atta_obj.pretrained_cp_smooth_scores[sample_idx].tolist(), 
                                              atta_obj.realtime_cp_include_larger_than[sample_idx],
                                              atta_obj.realtime_cp_pred_sets[sample_idx], 
                                              atta_obj.realtime_cp_smooth_scores[sample_idx].tolist()])


                ##### Store no labeled data #####
                for sample_idx in no_indices:
                    sample_level_list.append([idx, 
                                              batch.domain_name, 
                                              sample_idx, 
                                              "no", 
                                              realtime_cur_munch.numpy_pred_labels[sample_idx].item(),  
                                              true_labels[sample_idx].item(), 
                                              atta_obj.pretrained_cp_include_larger_than[sample_idx],
                                              atta_obj.pretrained_cp_pred_sets[sample_idx], 
                                              atta_obj.pretrained_cp_smooth_scores[sample_idx].tolist(), 
                                              atta_obj.realtime_cp_include_larger_than[sample_idx],
                                              atta_obj.realtime_cp_pred_sets[sample_idx], 
                                              atta_obj.realtime_cp_smooth_scores[sample_idx].tolist()])


                atta_obj.train_model()
                
                # append pseudo/oracle lr
                batch_level_new_dict['lr'] = atta_obj.lr
                batch_level_new_dict['oracle_lr'] = atta_obj.oracle_lr
                batch_level_new_dict['pseudo_lr'] = atta_obj.pseudo_lr
            
                # append 4 types of loss
                batch_level_new_dict['before_training_oracle_loss_sum'] = before_training_oracle_loss
                batch_level_new_dict['before_training_pseudo_loss_sum'] = before_training_pseudo_loss
                batch_level_new_dict['during_training_oracle_loss_avg'] = atta_obj.during_training_oracle_loss
                batch_level_new_dict['during_training_pseudo_loss_avg'] = atta_obj.during_training_pseudo_loss
                
                # append number of oracle data and pseudo data
                batch_level_new_dict['num_oracle_labeled'] = atta_obj.oracle_labeled_storage.num_elem()
                batch_level_new_dict['num_pseudo_labeled'] = atta_obj.pseudo_labeled_storage.num_elem()
                
                if atta_obj.pseudo_loss_coef is not None: batch_level_new_dict['pseudo_loss_coef'] = atta_obj.pseudo_loss_coef
                if atta_obj.oracle_loss_coef is not None: batch_level_new_dict['oracle_loss_coef'] = atta_obj.oracle_loss_coef


            batch_level_new_list = [batch_level_new_dict[d] for d in batch_level_columns]
            batch_level_list.append( batch_level_new_list )
            
            ########## print data ##########
            common_format = (
                "Batch Index: {:<4} -- "
                "Domain Name: {:<12} -- "
                "num_oracle_labeled: {:<8} -- "
                "num_pseudo_labeled: {:<8} -- "
                "pretrained_ILT_avg: {:<10} --"
                "realtime_ILT_avg: {:<10} --"
                "Acc: {:<4.3f} -- ")

            # Adjust float formatting only if needed
            if batch.domain_name != source_domain_name:
                pretrained_fmt = "{:<10.6f}"
                realtime_fmt = "{:<10.6f}"
            else:
                pretrained_fmt = "{:<10}"
                realtime_fmt = "{:<10}"

            # Final format string
            s = common_format.format(
                idx,
                batch.domain_name,
                batch_level_new_dict['num_oracle_labeled'],
                batch_level_new_dict['num_pseudo_labeled'],
                pretrained_fmt.format(batch_level_new_dict['pretrained_cp_include_larger_than_avg']),
                realtime_fmt.format(batch_level_new_dict['realtime_cp_include_larger_than_avg']),
                realtime_cur_munch.acc)

            print(s)

     
            # store pre-post information
            if idx != ( len(data_streams[data_stream_name]) - 1 ): # test if not the last batch
                next_domain_name = data_streams[data_stream_name][idx + 1].domain_name
                test_domain_name = batch.domain_name
                if (next_domain_name != test_domain_name) and (test_domain_name != source_domain_name): 
                    if data_stream_name == "domainwise":
                        for ele in domainwise_dataloaders:
                            inference_res = model_utils.model_inference( atta_obj.model, ele.dataloader, config.device )
                            continue_result_df.loc[ele.domain_name, test_domain_name] = inference_res.acc
                        continue_result_df.loc["realtime_acc", test_domain_name] = correct / num_samples
                        continue_result_df.loc["budgets", test_domain_name] = batch_level_new_list[-2]
                        print(f'\n{continue_result_df.round(4).to_markdown()}\n')
                    else:
                        for ele in domainwise_dataloaders:
                            inference_res = model_utils.model_inference( atta_obj.model, ele.dataloader, config.device )
                            random_result_df.loc[ele.domain_name, test_domain_name] = inference_res.acc
                        random_result_df.loc["realtime_acc", test_domain_name] = correct / num_samples
                        random_result_df.loc["budgets", test_domain_name] = batch_level_new_list[-2]
                        print(f'\n{random_result_df.round(4).to_markdown()}\n')
                    
                    correct = 0
                    num_samples = 0
            else: # test if the last batch
                test_domain_name = batch.domain_name
                if data_stream_name == "domainwise":
                        for ele in domainwise_dataloaders:
                            inference_res = model_utils.model_inference( atta_obj.model, ele.dataloader, config.device )
                            continue_result_df.loc[ele.domain_name, test_domain_name] = inference_res.acc
                        continue_result_df.loc["realtime_acc", test_domain_name] = correct / num_samples
                        continue_result_df.loc["budgets", test_domain_name] = batch_level_new_list[-2]
                        print(f'\n{continue_result_df.round(4).to_markdown()}\n')
                else:
                    for ele in domainwise_dataloaders:
                        inference_res = model_utils.model_inference( atta_obj.model, ele.dataloader, config.device )
                        random_result_df.loc[ele.domain_name, test_domain_name] = inference_res.acc
                    random_result_df.loc["realtime_acc", test_domain_name] = correct / num_samples
                    random_result_df.loc["budgets", test_domain_name] = batch_level_new_list[-2]
                    print(f'\n{random_result_df.round(4).to_markdown()}\n')
                
                correct = 0
                num_samples = 0


        # save dfs
        if save_csv:
            sname_ = data_stream_name + "_" + sname
            batch_level_df = pd.DataFrame(batch_level_list, columns=batch_level_columns)
            batch_level_df.to_csv(os.path.join( folder_path, sname_ + "_batch_level.csv"), index=False)
            sample_level_df = pd.DataFrame(sample_level_list, columns=sample_level_columns)
            sample_level_df.to_csv(os.path.join( folder_path, sname_ + "_sample_level.csv"), index=False)
            if data_stream_name == "domainwise":
                continue_result_df.to_csv(os.path.join( folder_path, sname_ + "_pre_post.csv"))
            else:
                random_result_df.to_csv(os.path.join( folder_path, sname_ + "_pre_post.csv"))

        print()

        atta_obj.destroy_model()
        del atta_obj


def qtc_metric_recorder(config, data_streams, domainwise_dataloaders, 
                          cal_dataloader, sname, folder_path, 
                          data_stream_names, domain_names, save_csv=True):

    # source domain name
    source_domain_name = domain_names[0]

    # store domainwise datastream results
    row_items = ['realtime_acc', 'budgets'] + domain_names
    col_items = domain_names
    continue_result_df = pd.DataFrame(index=row_items, columns=col_items, dtype=float)

    # store random datastream results
    row_items = ['realtime_acc', 'budgets'] + domain_names
    col_items = ['pre'] + [f"split_{i}" for i in range(4)] # modify here for different number of splits
    random_result_df = pd.DataFrame(index=row_items, columns=col_items, dtype=float)   

    # batch level columns
    batch_level_columns = ["batch_index", 
                           "batch_size", 
                           "domain_name", 
                           "realtime_model_acc", 

                           "pretrained_cp_cov", 
                           "pretrained_cp_pred_set_sizes_avg", 
                           "pretrained_cp_num_empty_sets", 
                           "pretrained_cp_include_larger_than_avg", 
                           
                           "realtime_cp_cov", 
                           "realtime_cp_pred_set_sizes_avg",
                           "realtime_cp_num_empty_sets", 
                           "realtime_cp_include_larger_than_avg",
                           
                           "oracle_lr",
                           "pseudo_lr",
                           "lr",
                           
                           "oracle_loss_coef",
                           "pseudo_loss_coef",
                           
                           "before_training_oracle_loss_sum",
                           "before_training_pseudo_loss_sum",
                           "during_training_oracle_loss_avg",
                           "during_training_pseudo_loss_avg",
                           
                           "num_oracle_labeled", 
                           "num_pseudo_labeled"]


    # sample level columns
    sample_level_columns = ["batch_id", 
                            "domain_name",
                            "sample_index", 
                            "labeling_method",
                            
                            "realtime_model_pred_label",
                            "true_label", 
                            
                            "pretrained_cp_include_larger_than",
                            "pretrained_cp_pred_set", 
                            "pretrained_cp_smooth_score", 
                            
                            "realtime_cp_include_larger_than",
                            "realtime_cp_pred_set", 
                            "realtime_cp_smooth_score",]


    for data_stream_name in data_stream_names:

        print(f"\n========== Working on {data_stream_name} ==========")

        atta_obj = register.atta_algs[config.atta.alg_name](config)
        
        print(f"========== atta alg name: {type(atta_obj).__name__} ==========")
        print(f"========== pretrained cp name: {type(atta_obj.pretrained_cp_predictor).__name__} ==========")
        print(f"========== realtime cp name: {type(atta_obj.pretrained_cp_predictor).__name__} ==========")

        # record pre-adaptation information
        if data_stream_name == "domainwise":
            for ele in domainwise_dataloaders:
                inference_res = model_utils.model_inference( atta_obj.model, ele.dataloader, config.device )
                continue_result_df.loc[ele.domain_name, source_domain_name] = inference_res.acc
            print(f'\n{continue_result_df.round(4).to_markdown()}\n')
        else:
            for ele in domainwise_dataloaders:
                inference_res = model_utils.model_inference( atta_obj.model, ele.dataloader, config.device )
                random_result_df.loc[ele.domain_name, "pre"] = inference_res.acc
            print(f'\n{random_result_df.round(4).to_markdown()}\n')


        batch_level_list = []
        sample_level_list = []
        
        # used for calculating accuracy whe changing domains
        correct = 0
        num_samples = 0

        for idx, batch in enumerate(data_streams[data_stream_name]):


            ########## Store batch level information ##########

            true_labels = data_utils.extract_labels(batch.dataloader)
            realtime_cur_munch = model_utils.model_inference( atta_obj.model, batch.dataloader, config.device )
            batch_size = len(batch.dataloader.dataset)


            batch_level_new_dict = {d:"NA" for d in batch_level_columns}
            batch_level_new_dict['batch_index'] = idx
            batch_level_new_dict['batch_size'] = batch_size
            batch_level_new_dict['domain_name'] = batch.domain_name
            batch_level_new_dict['realtime_model_acc'] = realtime_cur_munch.acc
            batch_level_new_dict['num_oracle_labeled'] = 0
            batch_level_new_dict['num_pseudo_labeled'] = 0
            
            if batch.domain_name != source_domain_name:
                
                correct += (realtime_cur_munch.acc * len(batch.dataloader.dataset))
                num_samples += len(batch.dataloader.dataset)

                # select data
                atta_obj.select_data(batch.dataloader, cal_dataloader)

                # get before training loss
                before_training_oracle_loss, before_training_pseudo_loss = atta_obj.get_before_training_loss()

                # get batch level metrics
                batch_level_new_dict = {d:"NA" for d in batch_level_columns}
                batch_level_new_dict['batch_index'] = idx
                batch_level_new_dict['batch_size'] = batch_size
                batch_level_new_dict['domain_name'] = batch.domain_name
                batch_level_new_dict['realtime_model_acc'] = realtime_cur_munch.acc
                
                batch_level_new_dict['pretrained_cp_cov'] = atta_obj.pretrained_cp_eval_res.cov
                batch_level_new_dict['pretrained_cp_pred_set_sizes_avg'] = atta_obj.pretrained_cp_eval_res.pred_set_size
                batch_level_new_dict['pretrained_cp_num_empty_sets'] = atta_obj.pretrained_cp_eval_res.num_empty_sets
                batch_level_new_dict['pretrained_cp_include_larger_than_avg'] = np.mean(atta_obj.pretrained_cp_include_larger_than).item()

                batch_level_new_dict['realtime_cp_cov'] = atta_obj.realtime_cp_eval_res.cov
                batch_level_new_dict['realtime_cp_pred_set_sizes_avg'] = atta_obj.realtime_cp_eval_res.pred_set_size
                batch_level_new_dict['realtime_cp_num_empty_sets'] = atta_obj.realtime_cp_eval_res.num_empty_sets
                batch_level_new_dict['realtime_cp_include_larger_than_avg'] = np.mean(atta_obj.realtime_cp_include_larger_than).item()


                oracle_indices = atta_obj.oracle_indices.tolist()
                pseudo_indices = atta_obj.pseudo_indices.tolist()
                temp = oracle_indices + pseudo_indices
                no_indices = [i for i in range(batch_size) if i not in temp]

                ##### Store oracle labeled data #####
                for sample_idx in oracle_indices:
                    sample_level_list.append([idx, 
                                              batch.domain_name, 
                                              sample_idx, 
                                              "oracle", 
                                              realtime_cur_munch.numpy_pred_labels[sample_idx].item(),  
                                              true_labels[sample_idx].item(), 
                                              atta_obj.pretrained_cp_include_larger_than[sample_idx],
                                              atta_obj.pretrained_cp_pred_sets[sample_idx], 
                                              atta_obj.pretrained_cp_smooth_scores[sample_idx].tolist(), 
                                              atta_obj.realtime_cp_include_larger_than[sample_idx],
                                              atta_obj.realtime_cp_pred_sets[sample_idx], 
                                              atta_obj.realtime_cp_smooth_scores[sample_idx].tolist()])
                    

                ##### Store pseudo labeled data #####
                for sample_idx in pseudo_indices:
                    sample_level_list.append([idx, 
                                              batch.domain_name, 
                                              sample_idx, 
                                              "pseudo", 
                                              realtime_cur_munch.numpy_pred_labels[sample_idx].item(),  
                                              true_labels[sample_idx].item(), 
                                              atta_obj.pretrained_cp_include_larger_than[sample_idx],
                                              atta_obj.pretrained_cp_pred_sets[sample_idx], 
                                              atta_obj.pretrained_cp_smooth_scores[sample_idx].tolist(), 
                                              atta_obj.realtime_cp_include_larger_than[sample_idx],
                                              atta_obj.realtime_cp_pred_sets[sample_idx], 
                                              atta_obj.realtime_cp_smooth_scores[sample_idx].tolist()])


                ##### Store no labeled data #####
                for sample_idx in no_indices:
                    sample_level_list.append([idx, 
                                              batch.domain_name, 
                                              sample_idx, 
                                              "no", 
                                              realtime_cur_munch.numpy_pred_labels[sample_idx].item(),  
                                              true_labels[sample_idx].item(), 
                                              atta_obj.pretrained_cp_include_larger_than[sample_idx],
                                              atta_obj.pretrained_cp_pred_sets[sample_idx], 
                                              atta_obj.pretrained_cp_smooth_scores[sample_idx].tolist(), 
                                              atta_obj.realtime_cp_include_larger_than[sample_idx],
                                              atta_obj.realtime_cp_pred_sets[sample_idx], 
                                              atta_obj.realtime_cp_smooth_scores[sample_idx].tolist()])


                atta_obj.train_model()
                
                # append pseudo/oracle lr
                batch_level_new_dict['lr'] = atta_obj.lr
                batch_level_new_dict['oracle_lr'] = atta_obj.oracle_lr
                batch_level_new_dict['pseudo_lr'] = atta_obj.pseudo_lr
            
                # append 4 types of loss
                batch_level_new_dict['before_training_oracle_loss_sum'] = before_training_oracle_loss
                batch_level_new_dict['before_training_pseudo_loss_sum'] = before_training_pseudo_loss
                batch_level_new_dict['during_training_oracle_loss_avg'] = atta_obj.during_training_oracle_loss
                batch_level_new_dict['during_training_pseudo_loss_avg'] = atta_obj.during_training_pseudo_loss
                
                # append number of oracle data and pseudo data
                batch_level_new_dict['num_oracle_labeled'] = atta_obj.oracle_labeled_storage.num_elem()
                batch_level_new_dict['num_pseudo_labeled'] = atta_obj.pseudo_labeled_storage.num_elem()
                
                if atta_obj.pseudo_loss_coef is not None: batch_level_new_dict['pseudo_loss_coef'] = atta_obj.pseudo_loss_coef
                if atta_obj.oracle_loss_coef is not None: batch_level_new_dict['oracle_loss_coef'] = atta_obj.oracle_loss_coef


            batch_level_new_list = [batch_level_new_dict[d] for d in batch_level_columns]
            batch_level_list.append( batch_level_new_list )
            
            ########## print data ##########
            common_format = (
                "Batch Index: {:<4} -- "
                "Domain Name: {:<12} -- "
                "num_oracle_labeled: {:<8} -- "
                "num_pseudo_labeled: {:<8} -- "
                "pretrained_ILT_avg: {:<10} --"
                "realtime_ILT_avg: {:<10} --"
                "Acc: {:<4.3f} -- ")

            # Adjust float formatting only if needed
            if batch.domain_name != source_domain_name:
                pretrained_fmt = "{:<10.6f}"
                realtime_fmt = "{:<10.6f}"
            else:
                pretrained_fmt = "{:<10}"
                realtime_fmt = "{:<10}"

            # Final format string
            s = common_format.format(
                idx,
                batch.domain_name,
                batch_level_new_dict['num_oracle_labeled'],
                batch_level_new_dict['num_pseudo_labeled'],
                pretrained_fmt.format(batch_level_new_dict['pretrained_cp_include_larger_than_avg']),
                realtime_fmt.format(batch_level_new_dict['realtime_cp_include_larger_than_avg']),
                realtime_cur_munch.acc)

            print(s)

     
            # store pre-post information
            if idx != ( len(data_streams[data_stream_name]) - 1 ): # test if not the last batch
                next_domain_name = data_streams[data_stream_name][idx + 1].domain_name
                test_domain_name = batch.domain_name
                if (next_domain_name != test_domain_name) and (test_domain_name != source_domain_name): 
                    if data_stream_name == "domainwise":
                        for ele in domainwise_dataloaders:
                            inference_res = model_utils.model_inference( atta_obj.model, ele.dataloader, config.device )
                            continue_result_df.loc[ele.domain_name, test_domain_name] = inference_res.acc
                        continue_result_df.loc["realtime_acc", test_domain_name] = correct / num_samples
                        continue_result_df.loc["budgets", test_domain_name] = batch_level_new_list[-2]
                        print(f'\n{continue_result_df.round(4).to_markdown()}\n')
                    else:
                        for ele in domainwise_dataloaders:
                            inference_res = model_utils.model_inference( atta_obj.model, ele.dataloader, config.device )
                            random_result_df.loc[ele.domain_name, test_domain_name] = inference_res.acc
                        random_result_df.loc["realtime_acc", test_domain_name] = correct / num_samples
                        random_result_df.loc["budgets", test_domain_name] = batch_level_new_list[-2]
                        print(f'\n{random_result_df.round(4).to_markdown()}\n')
                    
                    correct = 0
                    num_samples = 0
            else: # test if the last batch
                test_domain_name = batch.domain_name
                if data_stream_name == "domainwise":
                        for ele in domainwise_dataloaders:
                            inference_res = model_utils.model_inference( atta_obj.model, ele.dataloader, config.device )
                            continue_result_df.loc[ele.domain_name, test_domain_name] = inference_res.acc
                        continue_result_df.loc["realtime_acc", test_domain_name] = correct / num_samples
                        continue_result_df.loc["budgets", test_domain_name] = batch_level_new_list[-2]
                        print(f'\n{continue_result_df.round(4).to_markdown()}\n')
                else:
                    for ele in domainwise_dataloaders:
                        inference_res = model_utils.model_inference( atta_obj.model, ele.dataloader, config.device )
                        random_result_df.loc[ele.domain_name, test_domain_name] = inference_res.acc
                    random_result_df.loc["realtime_acc", test_domain_name] = correct / num_samples
                    random_result_df.loc["budgets", test_domain_name] = batch_level_new_list[-2]
                    print(f'\n{random_result_df.round(4).to_markdown()}\n')
                
                correct = 0
                num_samples = 0


        # save dfs
        if save_csv:
            sname_ = data_stream_name + "_" + sname
            batch_level_df = pd.DataFrame(batch_level_list, columns=batch_level_columns)
            batch_level_df.to_csv(os.path.join( folder_path, sname_ + "_batch_level.csv"), index=False)
            sample_level_df = pd.DataFrame(sample_level_list, columns=sample_level_columns)
            sample_level_df.to_csv(os.path.join( folder_path, sname_ + "_sample_level.csv"), index=False)
            if data_stream_name == "domainwise":
                continue_result_df.to_csv(os.path.join( folder_path, sname_ + "_pre_post.csv"))
            else:
                random_result_df.to_csv(os.path.join( folder_path, sname_ + "_pre_post.csv"))

        print()

        atta_obj.destroy_model()
        del atta_obj