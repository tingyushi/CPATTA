from cp_atta_package.utils import model_utils, data_utils
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

def cosine_similarity(A, B):
    # Step 1: Compute the norms (magnitudes) of each row in A and B
    A_norms = np.linalg.norm(A, axis=1, keepdims=True)
    B_norms = np.linalg.norm(B, axis=1, keepdims=True)
    
    # Step 2: Normalize the rows by dividing by their respective norms
    A_normalized = A / A_norms
    B_normalized = B / B_norms
    
    # Step 3: Compute the cosine similarity using matrix multiplication
    similarity = np.dot(A_normalized, B_normalized.T)
    
    return similarity



def fast_cosine_similarity(model, device, cal_dataloader, tta_dataloader):
    
    batch_size = cal_dataloader.batch_size

    grouped_images, grouped_labels = data_utils.collect_images_by_label(cal_dataloader)
    grouped_images_mean = torch.mean(grouped_images, dim=1)
    grouped_cal_dataloader =  DataLoader(TensorDataset(grouped_images_mean, grouped_labels), batch_size=batch_size, shuffle=False)
    grouped_cal_feats = model_utils.model_inference(model, grouped_cal_dataloader, device).numpy_feats
    tta_feats = model_utils.model_inference(model, tta_dataloader, device).numpy_feats

    assert tta_feats.shape[-1] == grouped_cal_feats.shape[-1]

    grouped_cos_similarity = cosine_similarity(tta_feats, grouped_cal_feats)
    assert grouped_cos_similarity.shape[0] == tta_feats.shape[0]
    assert grouped_cos_similarity.shape[1] == grouped_cal_feats.shape[0]
    assert np.all((grouped_cos_similarity >= -1) & (grouped_cos_similarity <= 1))

    res_arr = np.zeros((len(tta_dataloader.dataset) , len(cal_dataloader.dataset)))
    cal_labels = data_utils.extract_labels(cal_dataloader)
    for i in range(res_arr.shape[0]):
        for j in range(res_arr.shape[1]):
            res_arr[i, j] = grouped_cos_similarity[i , cal_labels[j].item()]
    
    assert np.all((res_arr >= -1) & (res_arr <= 1))

    return res_arr




def slow_cosine_similarity(model, device, cal_dataloader, tta_dataloader):
    
    batch_size = cal_dataloader.batch_size

    cal_feats = model_utils.model_inference(model, cal_dataloader, device).numpy_feats
    tta_feats = model_utils.model_inference(model, tta_dataloader, device).numpy_feats

    assert tta_feats.shape[-1] == cal_feats.shape[-1]

    cos_similarity = cosine_similarity(tta_feats, cal_feats)
    assert cos_similarity.shape[0] == tta_feats.shape[0]
    assert cos_similarity.shape[1] == cal_feats.shape[0]
    assert np.all((cos_similarity >= -1) & (cos_similarity <= 1))

    return cos_similarity



def domain_shift_signal(model, device, pre_dataloader, cur_dataloader):

    pre_feats = model_utils.model_inference(model, pre_dataloader, device).numpy_feats
    cur_feats = model_utils.model_inference(model, cur_dataloader, device).numpy_feats

    mean_pre_feats = np.mean(pre_feats, axis=0)
    mean_cur_feats = np.mean(cur_feats, axis=0)

    cos = np.dot(mean_cur_feats, mean_pre_feats) / (np.linalg.norm(mean_pre_feats) * np.linalg.norm(mean_cur_feats))

    return 1 - cos


def domain_shift_signal_using_feats(pre_feats, cur_feats):

    mean_pre_feats = np.mean(pre_feats, axis=0)
    mean_cur_feats = np.mean(cur_feats, axis=0)

    cos = np.dot(mean_cur_feats, mean_pre_feats) / (np.linalg.norm(mean_pre_feats) * np.linalg.norm(mean_cur_feats))

    return 1 - cos