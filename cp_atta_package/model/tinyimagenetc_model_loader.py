
from cp_atta_package.model.resnet_model import TinyImageNetCModel, ResNet
from cp_atta_package.utils.register import register
from cp_atta_package.utils import initialize


@register.model_load_func_register
def load_tinyimagenetc_model(config):
    return TinyImageNetCModel(config)