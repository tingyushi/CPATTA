import pandas as pd
import os
import numpy as np
import time
import sys
import itertools

from cp_atta_package.utils import process_ymal, model_utils, data_utils, pacs_metric_recorder
from cp_atta_package.utils.register import register
from cp_atta_package.model import pacs_model_loader
from cp_atta_package.data import pacs_data_loader
from cp_atta_package.atta_algs import cpatta, cpatta_base, cpatta_version5
from cp_atta_package.cp import nexcrc, qtc, exthr
import torch.multiprocessing
import re
import gc



if __name__ == "__main__":
    
    torch.multiprocessing.set_sharing_strategy('file_system')
    domains_complicated = ["photo", "art_painting", "cartoon", "sketch"]
    dataset_name = "pacs"

    # handle configs
    save_folder_path = ""
    cpatta_config = process_ymal.load_hyperparameters_as_munch("cpatta_config.yml")
    data_config = process_ymal.load_hyperparameters_as_munch(f"data_config.yml")

    data_streams, domainwise_dataloaders, cal_dataloader = register.data_loading_funcs[ f"load_{data_config.dataset.name.lower()}_data" ](data_config)

    pacs_metric_recorder.nexcrc_metric_recorder(config=cpatta_config, 
                                                data_streams=data_streams, 
                                                domainwise_dataloaders=domainwise_dataloaders, 
                                                cal_dataloader=cal_dataloader, 
                                                sname="cpatta", 
                                                folder_path=save_folder_path, 
                                                data_stream_names=["domainwise"], 
                                                domain_names=domains_complicated, save_csv=False)