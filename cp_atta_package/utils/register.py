r"""A kernel module that contains a global register for unified model, dataset, and OOD algorithms access.
"""


class Register(object):
    r"""
    Global register for unified model, dataset, and OOD algorithms access.
    """

    def __init__(self):
        self.model_load_funcs = dict()
        self.data_loading_funcs = dict()
        self.cp_predictors = dict()
        self.atta_algs = dict()
        self.nexcrc_loss_funcs = dict()

    def model_load_func_register(self, model_load_func):
        self.model_load_funcs[model_load_func.__name__] = model_load_func
        return model_load_func

    def data_load_func_register(self, data_load_func):
        self.data_loading_funcs[data_load_func.__name__] = data_load_func
        return data_load_func

    def cp_predictor_register(self, cp_predictor):
        self.cp_predictors[cp_predictor.__name__] = cp_predictor
        return cp_predictor

    def atta_alg_register(self, atta_alg):
        self.atta_algs[atta_alg.__name__] = atta_alg
        return atta_alg

    def nexcrc_loss_func_register(self, loss_func):
        self.nexcrc_loss_funcs[loss_func.__name__] = loss_func
        return loss_func



register = Register()  