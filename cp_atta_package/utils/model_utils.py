import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from munch import Munch

def model_inference(model: nn.Module, loader: DataLoader, device: torch.device):

    model.to(device) ; model.eval()
    correct = 0 ; total = 0
    torch_smx = [] ; torch_logits = []
    torch_pred_labels = []
    torch_feats = []


    with torch.no_grad():
        for x, y in loader:

            x = x.to(device)
            y = y.to(device)
            
            # inference
            feats = model.extract_feats(x)
            output = model.last_layer(feats)

            # collect data
            torch_feats.append( feats )
            torch_logits.append( output )
            torch_smx.append( torch.nn.functional.softmax( torch_logits[-1] , dim=-1) )
            torch_pred_labels.append( output.max(1)[1])
            
            # calculate accuracy
            correct += ( torch_pred_labels[-1] == y).float().sum()
            total += x.shape[0]
            
            

    acc = correct.item() / total

    torch_logits = torch.cat(torch_logits, dim=0)
    torch_smx = torch.cat(torch_smx, dim=0)
    torch_pred_labels = torch.cat(torch_pred_labels, dim=0)
    torch_feats = torch.cat(torch_feats, dim=0)

    numpy_logits = torch_logits.cpu().detach().numpy()
    numpy_smx = torch_smx.cpu().detach().numpy()
    numpy_pred_labels = torch_pred_labels.cpu().detach().numpy()
    numpy_feats = torch_feats.cpu().detach().numpy()

    return_value = {'acc': acc, 
                    'torch_logits': torch_logits, 
                    'torch_smx': torch_smx, 
                    'numpy_logits': numpy_logits, 
                    'numpy_smx': numpy_smx, 
                    "torch_pred_labels": torch_pred_labels, 
                    "numpy_pred_labels": numpy_pred_labels, 
                    "torch_feats": torch_feats, 
                    "numpy_feats": numpy_feats}

    return Munch(return_value)


def clean_accuracy(model: nn.Module,
                   loader: DataLoader,
                   device: torch.device):
    
    model.to(device) ; model.eval()
    correct = 0 ; total = 0
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            correct += (output.max(1)[1] == y).float().sum()
            total += x.shape[0]

    return correct.item() / total



def get_logits_smx(model: nn.Module, loader: DataLoader, device: torch.device):
    
    model.to(device) ; model.eval()
    
    torch_smx = [] ; torch_logits = []
    
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            torch_logits.append( model(x) )
            torch_smx.append( torch.nn.functional.softmax( torch_logits[-1] , dim=-1) )
    
    torch_logits = torch.cat(torch_logits, dim=0)
    torch_smx = torch.cat(torch_smx, dim=0)
    numpy_logits = torch_logits.cpu().detach().numpy()
    numpy_smx = torch_smx.cpu().detach().numpy()
    
    return torch_logits, torch_smx, numpy_logits, numpy_smx