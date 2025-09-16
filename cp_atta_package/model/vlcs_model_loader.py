from torchvision import transforms
from torchvision.datasets import ImageFolder


from cp_atta_package.model.resnet_model import VLCSModel, ResNet
from cp_atta_package.utils.register import register
from cp_atta_package.utils import initialize
from cp_atta_package.data import vlcs_data_loader

import torch
import os
import numpy as np
from sklearn.metrics import accuracy_score
from torch.nn.functional import cross_entropy


@register.model_load_func_register
def load_vlcs_model(config):
    return VLCSModel(config)