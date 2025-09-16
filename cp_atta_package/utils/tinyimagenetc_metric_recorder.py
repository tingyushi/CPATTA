import pandas as pd
import os
import numpy as np
import time
import sys
from torch.utils.data import TensorDataset, DataLoader

from cp_atta_package.utils import process_ymal, model_utils, data_utils
from cp_atta_package.utils.register import register
from cp_atta_package.model import tinyimagenetc_model_loader
from cp_atta_package.data import tinyimagenetc_data_loader
from cp_atta_package.atta_algs import cpatta
from cp_atta_package.cp import nexcrc, qtc, exthr



def nexcrc_metric_recorder(config, domainwise_dataloaders, cal_dataloader, 
                           sname, folder_path, domain_names, save_csv=True):

    # source domain name
    source_domain_name = domain_names[0]

    # store domainwise datastream results
    row_items = ['realtime_acc', 'budgets', 'Realtime_Coverage', 'Pretrained_Coverage', 'Oracle_Acc', 'Pseudo_Acc'] + domain_names
    col_items = domain_names
    continue_result_df = pd.DataFrame(index=row_items, columns=col_items, dtype=float)

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
                           "num_pseudo_labeled",
                           "current_batch_num_oracle_correct",
                           "current_batch_num_oracle_incorrect",
                           "current_batch_num_pseudo_correct",
                           "current_batch_num_pseudo_incorrect",]


    atta_obj = register.atta_algs[config.atta.alg_name](config)
    
    print(f"========== atta alg name: {type(atta_obj).__name__} ==========")
    print(f"========== pretrained cp name: {type(atta_obj.pretrained_cp_predictor).__name__} ==========")
    print(f"========== realtime cp name: {type(atta_obj.pretrained_cp_predictor).__name__} ==========")


    batch_level_list = []
    batch_idx = 0    

    for dmunch in domainwise_dataloaders:

        # used for calculating accuracy whe changing domains
        correct = 0
        realtime_covered = 0
        pretrained_covered = 0
        num_samples = 0
        oracle_correct = 0
        oracle_incorrect = 0
        pseudo_correct = 0
        pseudo_incorrect = 0


        for x, y in dmunch.dataloader:

            # move data to device
            x = x.to(config.device)
            y = y.to(config.device)

            # form dataloader without shuffling
            batch_dataloader = DataLoader(TensorDataset(x, y), batch_size=config.dataset.batch_size, shuffle=False)


            ########## Store batch level information ##########
            true_labels = data_utils.extract_labels(batch_dataloader)
            realtime_cur_munch = model_utils.model_inference( atta_obj.model, batch_dataloader, config.device )
            batch_size = len(batch_dataloader.dataset)


            batch_level_new_dict = {d:"NA" for d in batch_level_columns}
            batch_level_new_dict['batch_index'] = batch_idx
            batch_level_new_dict['batch_size'] = batch_size
            batch_level_new_dict['domain_name'] = dmunch.domain_name
            batch_level_new_dict['realtime_model_acc'] = realtime_cur_munch.acc
            batch_level_new_dict['num_oracle_labeled'] = 0
            batch_level_new_dict['num_pseudo_labeled'] = 0
            
            if dmunch.domain_name != source_domain_name:
                
                correct += (realtime_cur_munch.acc * len(batch_dataloader.dataset))
                num_samples += len(batch_dataloader.dataset)

                # select data
                atta_obj.select_data(batch_dataloader, cal_dataloader)

                # get before training loss
                before_training_oracle_loss, before_training_pseudo_loss = atta_obj.get_before_training_loss()

                # get batch level metrics
                batch_level_new_dict = {d:"NA" for d in batch_level_columns}
                batch_level_new_dict['batch_index'] = batch_idx
                batch_level_new_dict['batch_size'] = batch_size
                batch_level_new_dict['domain_name'] = dmunch.domain_name
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
                batch_level_new_dict['current_batch_num_oracle_correct'] = atta_obj.oracle_correct_num
                batch_level_new_dict['current_batch_num_oracle_incorrect'] = atta_obj.oracle_incorrect_num
                batch_level_new_dict['current_batch_num_pseudo_correct'] = atta_obj.pseudo_correct_num
                batch_level_new_dict['current_batch_num_pseudo_incorrect'] = atta_obj.pseudo_incorrect_num

                
                if atta_obj.pseudo_loss_coef is not None: batch_level_new_dict['pseudo_loss_coef'] = atta_obj.pseudo_loss_coef
                if atta_obj.oracle_loss_coef is not None: batch_level_new_dict['oracle_loss_coef'] = atta_obj.oracle_loss_coef
            
                # calculate coverage
                realtime_covered += batch_level_new_dict['realtime_cp_cov'] * len(batch_dataloader.dataset)
                pretrained_covered += batch_level_new_dict['pretrained_cp_cov'] * len(batch_dataloader.dataset)
                oracle_correct += batch_level_new_dict['current_batch_num_oracle_correct']
                oracle_incorrect += batch_level_new_dict['current_batch_num_oracle_incorrect']
                pseudo_correct += batch_level_new_dict['current_batch_num_pseudo_correct']
                pseudo_incorrect += batch_level_new_dict['current_batch_num_pseudo_incorrect']


            # increment batch index
            batch_idx += 1


            # append batch level information
            batch_level_new_list = [batch_level_new_dict[d] for d in batch_level_columns]
            batch_level_list.append( batch_level_new_list )
            

            ########## print data ##########
            common_format = (
                "Id: {:<4} -- "
                # "Domain Name: {:<20} -- "
                "Acc: {:<7.3f} -- "
                "oracle: {:<8} -- "
                "pseudo: {:<8} -- "
                "pt_ILT: {} --"
                "pt_cov: {} -- "
                "pt_pse_cov: {} -- "
                "pt_w: {} -- "
                "rt_ILT: {} --"
                "rt_cov: {} --"
                "rt_pse_cov: {} -- "
                "rt_w: {} -- "
                "oracle_lr: {:<7.3e} -- "
                "pseudo_lr: {:<7.3e} -- "
                 "alpha: {:<7.3f}"
                )

            # Adjust float formatting only if needed
            if dmunch.domain_name != source_domain_name:
                pretrained_weights_mean = "{:<10.3e}"
                realtime_weights_mean = "{:<10.3e}"
                pretrained_ilt = "{:<10.5f}"
                realtime_ilt = "{:<10.5f}"
                pretrained_cov = "{:<7.2f}"
                realtime_cov = "{:<7.2f}"
                pretrained_pse_cov = "{:<7.2f}"
                realtime_pse_cov = "{:<7.2f}"
            else:
                pretrained_weights_mean = "{:<10}"
                realtime_weights_mean = "{:<10}"
                pretrained_ilt = "{:<7}"
                realtime_ilt = "{:<7}"
                pretrained_cov = "{:<7}"
                realtime_cov = "{:<7}"
                pretrained_pse_cov = "{:<7}"
                realtime_pse_cov = "{:<7}"

            # Final format string
            s = common_format.format(
                batch_idx,
                # dmunch.domain_name,
                realtime_cur_munch.acc,
                batch_level_new_dict['num_oracle_labeled'],
                batch_level_new_dict['num_pseudo_labeled'],
                pretrained_ilt.format(batch_level_new_dict['pretrained_cp_include_larger_than_avg']),
                pretrained_cov.format( batch_level_new_dict['pretrained_cp_cov']),
                pretrained_pse_cov.format( batch_level_new_dict['pretrained_cp_pseudo_cov']),
                pretrained_weights_mean.format( batch_level_new_dict['pretrained_cp_weights_mean']),
                realtime_ilt.format(batch_level_new_dict['realtime_cp_include_larger_than_avg']),
                realtime_cov.format( batch_level_new_dict['realtime_cp_cov']),
                realtime_pse_cov.format( batch_level_new_dict['realtime_cp_pseudo_cov']),
                realtime_weights_mean.format( batch_level_new_dict['realtime_cp_weights_mean']),
                atta_obj.oracle_lr,
                atta_obj.pseudo_lr,
                atta_obj.config.cp.alpha
            )

            print(s)

        # store information
        if dmunch.domain_name != source_domain_name:
            continue_result_df.loc["realtime_acc", dmunch.domain_name] = correct / num_samples
            continue_result_df.loc["budgets",      dmunch.domain_name] = atta_obj.oracle_labeled_storage.num_elem()
            continue_result_df.loc["Realtime_Coverage",   dmunch.domain_name] = realtime_covered / num_samples
            continue_result_df.loc["Pretrained_Coverage", dmunch.domain_name] = pretrained_covered / num_samples
            continue_result_df.loc["Oracle_Acc", dmunch.domain_name] = oracle_correct / (oracle_correct + oracle_incorrect) if (oracle_correct + oracle_incorrect) > 0 else 0
            continue_result_df.loc["Pseudo_Acc", dmunch.domain_name] = pseudo_correct / (pseudo_correct + pseudo_incorrect) if (pseudo_correct + pseudo_incorrect) > 0 else 0


        # if last domain, test post adaptation acc for all domains
        print()
        if dmunch.domain_name == domain_names[-1]:
            for ele in domainwise_dataloaders:
                inference_res = model_utils.model_inference( atta_obj.model, ele.dataloader, config.device )
                continue_result_df.loc[ele.domain_name, dmunch.domain_name] = inference_res.acc
                print(f"Domain: {ele.domain_name}, Post Adaptation Acc: {inference_res.acc:.4f}")
        print()


        print(f'\n{continue_result_df.round(4).to_markdown()}\n')


    # save dfs
    if save_csv:
        sname_ = "domainwise" + "_" + sname
        batch_level_df = pd.DataFrame(batch_level_list, columns=batch_level_columns)
        batch_level_df.to_csv(os.path.join( folder_path, sname_ + "_batch_level.csv"), index=False)
        continue_result_df.to_csv(os.path.join( folder_path, sname_ + "_pre_post.csv"))
        
    atta_obj.destroy_model()
    del atta_obj






def exthr_metric_recorder(config, domainwise_dataloaders, cal_dataloader, 
                           sname, folder_path, domain_names, save_csv=True):

    # source domain name
    source_domain_name = domain_names[0]

    # store domainwise datastream results
    row_items = ['realtime_acc', 'budgets', 'Realtime_Coverage', 'Pretrained_Coverage', 'Oracle_Acc', 'Pseudo_Acc'] + domain_names
    col_items = domain_names
    continue_result_df = pd.DataFrame(index=row_items, columns=col_items, dtype=float)

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
                           "num_pseudo_labeled",
                           "current_batch_num_oracle_correct",
                           "current_batch_num_oracle_incorrect",
                           "current_batch_num_pseudo_correct",
                           "current_batch_num_pseudo_incorrect",]
    

    atta_obj = register.atta_algs[config.atta.alg_name](config)
    
    print(f"========== atta alg name: {type(atta_obj).__name__} ==========")
    print(f"========== pretrained cp name: {type(atta_obj.pretrained_cp_predictor).__name__} ==========")
    print(f"========== realtime cp name: {type(atta_obj.pretrained_cp_predictor).__name__} ==========")


    batch_level_list = []
    batch_idx = 0    

    for dmunch in domainwise_dataloaders:

        # used for calculating accuracy whe changing domains
        correct = 0
        realtime_covered = 0
        pretrained_covered = 0
        num_samples = 0
        oracle_correct = 0
        oracle_incorrect = 0
        pseudo_correct = 0
        pseudo_incorrect = 0


        for x, y in dmunch.dataloader:

            # move data to device
            x = x.to(config.device)
            y = y.to(config.device)

            # form dataloader without shuffling
            batch_dataloader = DataLoader(TensorDataset(x, y), batch_size=config.dataset.batch_size, shuffle=False)


            ########## Store batch level information ##########
            true_labels = data_utils.extract_labels(batch_dataloader)
            realtime_cur_munch = model_utils.model_inference( atta_obj.model, batch_dataloader, config.device )
            batch_size = len(batch_dataloader.dataset)


            batch_level_new_dict = {d:"NA" for d in batch_level_columns}
            batch_level_new_dict['batch_index'] = batch_idx
            batch_level_new_dict['batch_size'] = batch_size
            batch_level_new_dict['domain_name'] = dmunch.domain_name
            batch_level_new_dict['realtime_model_acc'] = realtime_cur_munch.acc
            batch_level_new_dict['num_oracle_labeled'] = 0
            batch_level_new_dict['num_pseudo_labeled'] = 0
            
            if dmunch.domain_name != source_domain_name:
                
                correct += (realtime_cur_munch.acc * len(batch_dataloader.dataset))
                num_samples += len(batch_dataloader.dataset)

                # select data
                atta_obj.select_data(batch_dataloader, cal_dataloader)

                # get before training loss
                before_training_oracle_loss, before_training_pseudo_loss = atta_obj.get_before_training_loss()

                # get batch level metrics
                batch_level_new_dict = {d:"NA" for d in batch_level_columns}
                batch_level_new_dict['batch_index'] = batch_idx
                batch_level_new_dict['batch_size'] = batch_size
                batch_level_new_dict['domain_name'] = dmunch.domain_name
                batch_level_new_dict['realtime_model_acc'] = realtime_cur_munch.acc

                
                batch_level_new_dict['pretrained_cp_cov'] = atta_obj.pretrained_cp_eval_res.cov
                batch_level_new_dict['pretrained_cp_pred_set_sizes_avg'] = atta_obj.pretrained_cp_eval_res.pred_set_size
                batch_level_new_dict['pretrained_cp_num_empty_sets'] = atta_obj.pretrained_cp_eval_res.num_empty_sets
                batch_level_new_dict['pretrained_cp_include_larger_than_avg'] = np.mean(atta_obj.pretrained_cp_include_larger_than).item()


                batch_level_new_dict['realtime_cp_cov'] = atta_obj.realtime_cp_eval_res.cov
                batch_level_new_dict['realtime_cp_pred_set_sizes_avg'] = atta_obj.realtime_cp_eval_res.pred_set_size
                batch_level_new_dict['realtime_cp_num_empty_sets'] = atta_obj.realtime_cp_eval_res.num_empty_sets
                batch_level_new_dict['realtime_cp_include_larger_than_avg'] = np.mean(atta_obj.realtime_cp_include_larger_than).item()

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
                batch_level_new_dict['current_batch_num_oracle_correct'] = atta_obj.oracle_correct_num
                batch_level_new_dict['current_batch_num_oracle_incorrect'] = atta_obj.oracle_incorrect_num
                batch_level_new_dict['current_batch_num_pseudo_correct'] = atta_obj.pseudo_correct_num
                batch_level_new_dict['current_batch_num_pseudo_incorrect'] = atta_obj.pseudo_incorrect_num

                
                if atta_obj.pseudo_loss_coef is not None: batch_level_new_dict['pseudo_loss_coef'] = atta_obj.pseudo_loss_coef
                if atta_obj.oracle_loss_coef is not None: batch_level_new_dict['oracle_loss_coef'] = atta_obj.oracle_loss_coef
            
                # calculate coverage
                realtime_covered += batch_level_new_dict['realtime_cp_cov'] * len(batch_dataloader.dataset)
                pretrained_covered += batch_level_new_dict['pretrained_cp_cov'] * len(batch_dataloader.dataset)
                oracle_correct += batch_level_new_dict['current_batch_num_oracle_correct']
                oracle_incorrect += batch_level_new_dict['current_batch_num_oracle_incorrect']
                pseudo_correct += batch_level_new_dict['current_batch_num_pseudo_correct']
                pseudo_incorrect += batch_level_new_dict['current_batch_num_pseudo_incorrect']


            # increment batch index
            batch_idx += 1


            # append batch level information
            batch_level_new_list = [batch_level_new_dict[d] for d in batch_level_columns]
            batch_level_list.append( batch_level_new_list )
            

            ########## print data ##########
            common_format = (
                "Id: {:<4} -- "
                # "Domain Name: {:<20} -- "
                "Acc: {:<7.3f} -- "
                "oracle: {:<8} -- "
                "pseudo: {:<8} -- "
                "pt_ILT: {} --"
                "pt_cov: {} -- "
                "rt_ILT: {} --"
                "rt_cov: {} --"
                "oracle_lr: {:<7.3e} -- "
                "pseudo_lr: {:<7.3e} -- "
                "alpha: {:<7.3f}"
                )

            # Adjust float formatting only if needed
            if dmunch.domain_name != source_domain_name:
                pretrained_ilt = "{:<10.5f}"
                realtime_ilt = "{:<10.5f}"
                pretrained_cov = "{:<7.2f}"
                realtime_cov = "{:<7.2f}"
            else:
                pretrained_ilt = "{:<7}"
                realtime_ilt = "{:<7}"
                pretrained_cov = "{:<7}"
                realtime_cov = "{:<7}"
                
            # Final format string
            s = common_format.format(
                batch_idx,
                # dmunch.domain_name,
                realtime_cur_munch.acc,
                batch_level_new_dict['num_oracle_labeled'],
                batch_level_new_dict['num_pseudo_labeled'],
                pretrained_ilt.format(batch_level_new_dict['pretrained_cp_include_larger_than_avg']),
                pretrained_cov.format( batch_level_new_dict['pretrained_cp_cov']),
                realtime_ilt.format(batch_level_new_dict['realtime_cp_include_larger_than_avg']),
                realtime_cov.format( batch_level_new_dict['realtime_cp_cov']),
                atta_obj.oracle_lr,
                atta_obj.pseudo_lr,
                atta_obj.config.cp.alpha
            )

            print(s)

        # store information
        if dmunch.domain_name != source_domain_name:
            continue_result_df.loc["realtime_acc", dmunch.domain_name] = correct / num_samples
            continue_result_df.loc["budgets",      dmunch.domain_name] = atta_obj.oracle_labeled_storage.num_elem()
            continue_result_df.loc["Realtime_Coverage",   dmunch.domain_name] = realtime_covered / num_samples
            continue_result_df.loc["Pretrained_Coverage", dmunch.domain_name] = pretrained_covered / num_samples
            continue_result_df.loc["Oracle_Acc", dmunch.domain_name] = oracle_correct / (oracle_correct + oracle_incorrect) if (oracle_correct + oracle_incorrect) > 0 else 0
            continue_result_df.loc["Pseudo_Acc", dmunch.domain_name] = pseudo_correct / (pseudo_correct + pseudo_incorrect) if (pseudo_correct + pseudo_incorrect) > 0 else 0


        # if last domain, test post adaptation acc for all domains
        print()
        if dmunch.domain_name == domain_names[-1]:
            for ele in domainwise_dataloaders:
                inference_res = model_utils.model_inference( atta_obj.model, ele.dataloader, config.device )
                continue_result_df.loc[ele.domain_name, dmunch.domain_name] = inference_res.acc
                print(f"Domain: {ele.domain_name}, Post Adaptation Acc: {inference_res.acc:.4f}")
        print()


        print(f'\n{continue_result_df.round(4).to_markdown()}\n')


    # save dfs
    if save_csv:
        sname_ = "domainwise" + "_" + sname
        batch_level_df = pd.DataFrame(batch_level_list, columns=batch_level_columns)
        batch_level_df.to_csv(os.path.join( folder_path, sname_ + "_batch_level.csv"), index=False)
        continue_result_df.to_csv(os.path.join( folder_path, sname_ + "_pre_post.csv"))
        
    atta_obj.destroy_model()
    del atta_obj





def qtc_metric_recorder(config, domainwise_dataloaders, cal_dataloader, 
                           sname, folder_path, domain_names, save_csv=True):

    # source domain name
    source_domain_name = domain_names[0]

    # store domainwise datastream results
    row_items = ['realtime_acc', 'budgets', 'Realtime_Coverage', 'Pretrained_Coverage', 'Oracle_Acc', 'Pseudo_Acc'] + domain_names
    col_items = domain_names
    continue_result_df = pd.DataFrame(index=row_items, columns=col_items, dtype=float)

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
                           "num_pseudo_labeled",
                           "current_batch_num_oracle_correct",
                           "current_batch_num_oracle_incorrect",
                           "current_batch_num_pseudo_correct",
                           "current_batch_num_pseudo_incorrect",]


    atta_obj = register.atta_algs[config.atta.alg_name](config)
    
    print(f"========== atta alg name: {type(atta_obj).__name__} ==========")
    print(f"========== pretrained cp name: {type(atta_obj.pretrained_cp_predictor).__name__} ==========")
    print(f"========== realtime cp name: {type(atta_obj.pretrained_cp_predictor).__name__} ==========")

    batch_level_list = []
    batch_idx = 0    

    for dmunch in domainwise_dataloaders:

        # used for calculating accuracy whe changing domains
        correct = 0
        realtime_covered = 0
        pretrained_covered = 0
        num_samples = 0
        oracle_correct = 0
        oracle_incorrect = 0
        pseudo_correct = 0
        pseudo_incorrect = 0


        for x, y in dmunch.dataloader:

            # move data to device
            x = x.to(config.device)
            y = y.to(config.device)

            # form dataloader without shuffling
            batch_dataloader = DataLoader(TensorDataset(x, y), batch_size=config.dataset.batch_size, shuffle=False)


            ########## Store batch level information ##########
            true_labels = data_utils.extract_labels(batch_dataloader)
            realtime_cur_munch = model_utils.model_inference( atta_obj.model, batch_dataloader, config.device )
            batch_size = len(batch_dataloader.dataset)


            batch_level_new_dict = {d:"NA" for d in batch_level_columns}
            batch_level_new_dict['batch_index'] = batch_idx
            batch_level_new_dict['batch_size'] = batch_size
            batch_level_new_dict['domain_name'] = dmunch.domain_name
            batch_level_new_dict['realtime_model_acc'] = realtime_cur_munch.acc
            batch_level_new_dict['num_oracle_labeled'] = 0
            batch_level_new_dict['num_pseudo_labeled'] = 0
            
            if dmunch.domain_name != source_domain_name:
                
                correct += (realtime_cur_munch.acc * len(batch_dataloader.dataset))
                num_samples += len(batch_dataloader.dataset)

                # select data
                atta_obj.select_data(batch_dataloader, cal_dataloader)

                # get before training loss
                before_training_oracle_loss, before_training_pseudo_loss = atta_obj.get_before_training_loss()

                # get batch level metrics
                batch_level_new_dict = {d:"NA" for d in batch_level_columns}
                batch_level_new_dict['batch_index'] = batch_idx
                batch_level_new_dict['batch_size'] = batch_size
                batch_level_new_dict['domain_name'] = dmunch.domain_name
                batch_level_new_dict['realtime_model_acc'] = realtime_cur_munch.acc

                
                batch_level_new_dict['pretrained_cp_cov'] = atta_obj.pretrained_cp_eval_res.cov
                batch_level_new_dict['pretrained_cp_pred_set_sizes_avg'] = atta_obj.pretrained_cp_eval_res.pred_set_size
                batch_level_new_dict['pretrained_cp_num_empty_sets'] = atta_obj.pretrained_cp_eval_res.num_empty_sets
                batch_level_new_dict['pretrained_cp_include_larger_than_avg'] = np.mean(atta_obj.pretrained_cp_include_larger_than).item()


                batch_level_new_dict['realtime_cp_cov'] = atta_obj.realtime_cp_eval_res.cov
                batch_level_new_dict['realtime_cp_pred_set_sizes_avg'] = atta_obj.realtime_cp_eval_res.pred_set_size
                batch_level_new_dict['realtime_cp_num_empty_sets'] = atta_obj.realtime_cp_eval_res.num_empty_sets
                batch_level_new_dict['realtime_cp_include_larger_than_avg'] = np.mean(atta_obj.realtime_cp_include_larger_than).item()

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
                batch_level_new_dict['current_batch_num_oracle_correct'] = atta_obj.oracle_correct_num
                batch_level_new_dict['current_batch_num_oracle_incorrect'] = atta_obj.oracle_incorrect_num
                batch_level_new_dict['current_batch_num_pseudo_correct'] = atta_obj.pseudo_correct_num
                batch_level_new_dict['current_batch_num_pseudo_incorrect'] = atta_obj.pseudo_incorrect_num

                
                if atta_obj.pseudo_loss_coef is not None: batch_level_new_dict['pseudo_loss_coef'] = atta_obj.pseudo_loss_coef
                if atta_obj.oracle_loss_coef is not None: batch_level_new_dict['oracle_loss_coef'] = atta_obj.oracle_loss_coef
            
                # calculate coverage
                realtime_covered += batch_level_new_dict['realtime_cp_cov'] * len(batch_dataloader.dataset)
                pretrained_covered += batch_level_new_dict['pretrained_cp_cov'] * len(batch_dataloader.dataset)
                oracle_correct += batch_level_new_dict['current_batch_num_oracle_correct']
                oracle_incorrect += batch_level_new_dict['current_batch_num_oracle_incorrect']
                pseudo_correct += batch_level_new_dict['current_batch_num_pseudo_correct']
                pseudo_incorrect += batch_level_new_dict['current_batch_num_pseudo_incorrect']


            # increment batch index
            batch_idx += 1


            # append batch level information
            batch_level_new_list = [batch_level_new_dict[d] for d in batch_level_columns]
            batch_level_list.append( batch_level_new_list )
            

            ########## print data ##########
            common_format = (
                "Id: {:<4} -- "
                # "Domain Name: {:<20} -- "
                "Acc: {:<7.3f} -- "
                "oracle: {:<8} -- "
                "pseudo: {:<8} -- "
                "pt_ILT: {} --"
                "pt_cov: {} -- "
                "rt_ILT: {} --"
                "rt_cov: {} --"
                "oracle_lr: {:<7.3e} -- "
                "pseudo_lr: {:<7.3e} -- "
                "alpha: {:<7.3f}"
                )

            # Adjust float formatting only if needed
            if dmunch.domain_name != source_domain_name:
                pretrained_ilt = "{:<10.5f}"
                realtime_ilt = "{:<10.5f}"
                pretrained_cov = "{:<7.2f}"
                realtime_cov = "{:<7.2f}"
            else:
                pretrained_ilt = "{:<7}"
                realtime_ilt = "{:<7}"
                pretrained_cov = "{:<7}"
                realtime_cov = "{:<7}"
                
            # Final format string
            s = common_format.format(
                batch_idx,
                # dmunch.domain_name,
                realtime_cur_munch.acc,
                batch_level_new_dict['num_oracle_labeled'],
                batch_level_new_dict['num_pseudo_labeled'],
                pretrained_ilt.format(batch_level_new_dict['pretrained_cp_include_larger_than_avg']),
                pretrained_cov.format( batch_level_new_dict['pretrained_cp_cov']),
                realtime_ilt.format(batch_level_new_dict['realtime_cp_include_larger_than_avg']),
                realtime_cov.format( batch_level_new_dict['realtime_cp_cov']),
                atta_obj.oracle_lr,
                atta_obj.pseudo_lr,
                atta_obj.config.cp.alpha
            )

            print(s)

        # store information
        if dmunch.domain_name != source_domain_name:
            continue_result_df.loc["realtime_acc", dmunch.domain_name] = correct / num_samples
            continue_result_df.loc["budgets",      dmunch.domain_name] = atta_obj.oracle_labeled_storage.num_elem()
            continue_result_df.loc["Realtime_Coverage",   dmunch.domain_name] = realtime_covered / num_samples
            continue_result_df.loc["Pretrained_Coverage", dmunch.domain_name] = pretrained_covered / num_samples
            continue_result_df.loc["Oracle_Acc", dmunch.domain_name] = oracle_correct / (oracle_correct + oracle_incorrect) if (oracle_correct + oracle_incorrect) > 0 else 0
            continue_result_df.loc["Pseudo_Acc", dmunch.domain_name] = pseudo_correct / (pseudo_correct + pseudo_incorrect) if (pseudo_correct + pseudo_incorrect) > 0 else 0


        # if last domain, test post adaptation acc for all domains
        print()
        if dmunch.domain_name == domain_names[-1]:
            for ele in domainwise_dataloaders:
                inference_res = model_utils.model_inference( atta_obj.model, ele.dataloader, config.device )
                continue_result_df.loc[ele.domain_name, dmunch.domain_name] = inference_res.acc
                print(f"Domain: {ele.domain_name}, Post Adaptation Acc: {inference_res.acc:.4f}")
        print()


        print(f'\n{continue_result_df.round(4).to_markdown()}\n')


    # save dfs
    if save_csv:
        sname_ = "domainwise" + "_" + sname
        batch_level_df = pd.DataFrame(batch_level_list, columns=batch_level_columns)
        batch_level_df.to_csv(os.path.join( folder_path, sname_ + "_batch_level.csv"), index=False)
        continue_result_df.to_csv(os.path.join( folder_path, sname_ + "_pre_post.csv"))
        
    atta_obj.destroy_model()
    del atta_obj