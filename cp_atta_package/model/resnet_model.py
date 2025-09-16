import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import os
from os.path import join as opj

class Identity(nn.Module):
    """An identity layer"""
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class ResNet(torch.nn.Module):
    """ResNet with the softmax chopped off and the batchnorm frozen"""
    def __init__(self, hparams):
        super(ResNet, self).__init__()
        input_shape = hparams.model.input_shape
        if hparams.model.resnet18:
            self.network = torchvision.models.resnet18(weights='DEFAULT')
            self.n_outputs = 512
        else:
            self.network = torchvision.models.resnet50(weights='DEFAULT')
            self.n_outputs = 2048

        # self.network = remove_batch_norm_from_resnet(self.network)

        # adapt number of channels
        nc = input_shape[0]
        if nc != 3:
            tmp = self.network.conv1.weight.data.clone()

            self.network.conv1 = nn.Conv2d(
                nc, 64, kernel_size=(7, 7),
                stride=(2, 2), padding=(3, 3), bias=False)

            for i in range(nc):
                self.network.conv1.weight.data[:, i, :, :] = tmp[:, i % 3, :, :]

        # save memory
        # self.fc = copy.deepcopy(self.network.fc)
        del self.network.fc
        self.network.fc = Identity()

        self.fr_bn = hparams.model.freeze_bn
        self.freeze_bn()
        self.hparams = hparams
        self.dropout = nn.Dropout(hparams.model.dropout_rate)

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        return self.dropout(self.network(x))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        self.freeze_bn()

    def freeze_bn(self):
        if self.fr_bn:
            for m in self.network.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()



class PACSModel(torch.nn.Module):

    def __init__(self, config):
        super(PACSModel, self).__init__()
        
        self.encoder = ResNet(config)
        self.fc = torch.nn.Linear(self.encoder.n_outputs, config.dataset.num_classes)
        
        assert os.path.exists( opj( config.model.path, config.dataset.name.lower() ) )

        if config.model.type == "provided":

            print(f"========== Loading Provided {config.dataset.name.upper()} Model ==========")
            path = os.path.join(config.model.pacs_encoder_ckpt_path , f"pacs_pretrained_encoder_provided.pth")
            self.encoder.load_state_dict(torch.load(path, map_location=config.device), strict=False)
            path = os.path.join(config.model.pacs_fc_ckpt_path , f"pacs_pretrained_fc_provided.pth")
            self.fc.load_state_dict(torch.load(path, map_location=config.device))

        elif config.model.type == "trained":
            
            print(f"========== Loading Trained {config.dataset.name.upper()} Model ==========")
            path = os.path.join(config.model.pacs_encoder_ckpt_path , f"pacs_pretrained_encoder_trained.pth")
            self.encoder.load_state_dict(torch.load(path, map_location=config.device), strict=False)
            path = os.path.join(config.model.pacs_fc_ckpt_path , f"pacs_pretrained_fc_trained.pth")
            self.fc.load_state_dict(torch.load(path, map_location=config.device))

        elif config.model.type == "original":
            
            print(f"========== Returning an untrained {config.dataset.name.upper()} model")

        elif config.model.type == "allsource":
            
            print(f"========== Loading {config.dataset.name.upper()} Model trained with all source ==========")
            path = os.path.join(config.model.pacs_encoder_ckpt_path , f"pacs_pretrained_encoder_allsource.pth")
            self.encoder.load_state_dict(torch.load(path, map_location=config.device), strict=False)
            path = os.path.join(config.model.pacs_fc_ckpt_path , f"pacs_pretrained_fc_allsource.pth")
            self.fc.load_state_dict(torch.load(path, map_location=config.device))

        elif config.model.type == "partialsource":

            print(f"========== Loading {config.dataset.name.upper()} Model trained with partial source model ==========")

            path = opj( config.model.path, 
                        config.dataset.name.lower(),
                        f"{ config.dataset.name.lower() }_pretrained_encoder_partialsource_seed_{config.seed}.pth")
    

            self.encoder.load_state_dict(torch.load(path, map_location=config.device), strict=False)
    
            path = opj( config.model.path, 
                        config.dataset.name.lower(),
                        f"{ config.dataset.name.lower() }_pretrained_fc_partialsource_seed_{config.seed}.pth")
    
            self.fc.load_state_dict(torch.load(path, map_location=config.device))
            
        else:
            assert False

    
    def forward(self, x):
        return self.fc(self.encoder(x))

    def extract_feats(self, x):
        out = self.encoder(x)
        return out
    
    def last_layer(self, x):
        return self.fc(x)
    




class VLCSModel(torch.nn.Module):

    def __init__(self, config):
        super(VLCSModel, self).__init__()

        self.encoder = ResNet(config)
        self.fc = torch.nn.Linear(self.encoder.n_outputs, config.dataset.num_classes)

        assert os.path.exists( opj( config.model.path, config.dataset.name.lower() ) )

        if config.model.type == "provided":

            print(f"========== Loading Provided { config.dataset.name.upper() } Model ==========")
            path = os.path.join(config.model.vlcs_encoder_ckpt_path , f"vlcs_pretrained_encoder_provided.pth")
            self.encoder.load_state_dict(torch.load(path, map_location=config.device), strict=False)
            path = os.path.join(config.model.vlcs_fc_ckpt_path , f"vlcs_pretrained_fc_provided.pth")
            self.fc.load_state_dict(torch.load(path, map_location=config.device))

        elif config.model.type == "trained":

            print(f"========== Loading Trained { config.dataset.name.upper() } Model ==========")
            path = os.path.join(config.model.vlcs_encoder_ckpt_path , f"vlcs_pretrained_encoder_trained.pth")
            self.encoder.load_state_dict(torch.load(path, map_location=config.device), strict=False)
            path = os.path.join(config.model.vlcs_fc_ckpt_path , f"vlcs_pretrained_fc_trained.pth")
            self.fc.load_state_dict(torch.load(path, map_location=config.device))

        elif config.model.type == "original":
            
            print(f"========== Returning an untrained { config.dataset.name.upper() } model")

        elif config.model.type == "allsource":

            print(f"========== Loading { config.dataset.name.upper() } Model trained with all source ==========")
            path = os.path.join(config.model.vlcs_encoder_ckpt_path , f"vlcs_pretrained_encoder_allsource.pth")
            self.encoder.load_state_dict(torch.load(path, map_location=config.device), strict=False)
            path = os.path.join(config.model.vlcs_fc_ckpt_path , f"vlcs_pretrained_fc_allsource.pth")
            self.fc.load_state_dict(torch.load(path, map_location=config.device))

        elif config.model.type == "partialsource":

            print(f"========== Loading { config.dataset.name.upper() } Model trained with partial source model ==========")
            path = opj (config.model.path, 
                        config.dataset.name.lower(),
                        f"{ config.dataset.name.lower() }_pretrained_encoder_partialsource_seed_{config.seed}.pth")
            
            self.encoder.load_state_dict(torch.load(path, map_location=config.device), strict=False)
            
            path = opj (config.model.path, 
                        config.dataset.name.lower(),
                        f"{ config.dataset.name.lower() }_pretrained_fc_partialsource_seed_{config.seed}.pth")
            
            self.fc.load_state_dict(torch.load(path, map_location=config.device))
            
        else:
            assert False

    
    def forward(self, x):
        return self.fc(self.encoder(x))

    def extract_feats(self, x):
        out = self.encoder(x)
        return out
    
    def last_layer(self, x):
        return self.fc(x)



class TinyImageNetCModel(torch.nn.Module):

    def __init__(self, config):
        super(TinyImageNetCModel, self).__init__()

        self.encoder = ResNet(config)
        self.fc = torch.nn.Linear(self.encoder.n_outputs, config.dataset.num_classes)

        assert os.path.exists( opj( config.model.path, config.dataset.name.lower() ) )

        if config.model.type == "provided":

            print(f"========== Loading Provided { config.dataset.name.upper() } Model ==========")
            path = os.path.join(config.model.vlcs_encoder_ckpt_path , f"vlcs_pretrained_encoder_provided.pth")
            self.encoder.load_state_dict(torch.load(path, map_location=config.device), strict=False)
            path = os.path.join(config.model.vlcs_fc_ckpt_path , f"vlcs_pretrained_fc_provided.pth")
            self.fc.load_state_dict(torch.load(path, map_location=config.device))

        elif config.model.type == "trained":

            print(f"========== Loading Trained { config.dataset.name.upper() } Model ==========")
            path = os.path.join(config.model.vlcs_encoder_ckpt_path , f"vlcs_pretrained_encoder_trained.pth")
            self.encoder.load_state_dict(torch.load(path, map_location=config.device), strict=False)
            path = os.path.join(config.model.vlcs_fc_ckpt_path , f"vlcs_pretrained_fc_trained.pth")
            self.fc.load_state_dict(torch.load(path, map_location=config.device))

        elif config.model.type == "original":
            
            print(f"========== Returning an untrained { config.dataset.name.upper() } model")

        elif config.model.type == "allsource":

            print(f"========== Loading { config.dataset.name.upper() } Model trained with all source ==========")
            path = os.path.join(config.model.vlcs_encoder_ckpt_path , f"vlcs_pretrained_encoder_allsource.pth")
            self.encoder.load_state_dict(torch.load(path, map_location=config.device), strict=False)
            path = os.path.join(config.model.vlcs_fc_ckpt_path , f"vlcs_pretrained_fc_allsource.pth")
            self.fc.load_state_dict(torch.load(path, map_location=config.device))

        elif config.model.type == "partialsource":

            print(f"========== Loading { config.dataset.name.upper() } Model trained with partial source model ==========")
            path = opj (config.model.path, 
                        config.dataset.name.lower(),
                        f"{ config.dataset.name.lower() }_pretrained_encoder_partialsource_seed_{config.seed}.pth")
            
            self.encoder.load_state_dict(torch.load(path, map_location=config.device), strict=False)
            
            path = opj (config.model.path, 
                        config.dataset.name.lower(),
                        f"{ config.dataset.name.lower() }_pretrained_fc_partialsource_seed_{config.seed}.pth")
            
            self.fc.load_state_dict(torch.load(path, map_location=config.device))
            
        else:
            assert False

    
    def forward(self, x):
        return self.fc(self.encoder(x))

    def extract_feats(self, x):
        out = self.encoder(x)
        return out
    
    def last_layer(self, x):
        return self.fc(x)
    