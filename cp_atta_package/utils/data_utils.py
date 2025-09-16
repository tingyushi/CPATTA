import numpy as np
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
import copy
import torch
from munch import Munch
from collections import defaultdict


def extract_labels(loader: DataLoader):
    all_labels_np = np.concatenate([batch_labels.cpu().detach().numpy() for _, batch_labels in loader])
    return all_labels_np


def extract_data(loader: DataLoader):
    all_data_np = np.concatenate([batch_data.cpu().detach().numpy() for batch_data, _ in loader])
    return all_data_np


def batchify_data(data, labels, batch_size, num_of_batches, domain_name):
    
    assert data.shape[0] >= batch_size * num_of_batches
    
    # handle source data
    indices = np.random.choice(data.shape[0], batch_size*num_of_batches, replace=False)
    data_batches = data[indices]
    labels_batches = labels[indices]
    dataset = TensorDataset(data_batches, labels_batches)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)    
    
    res = []
    for x, y in dataloader:
        temp_dataset = TensorDataset(x, y)
        temp_dataloader = DataLoader(temp_dataset, batch_size=batch_size, shuffle=False)   
        dic = {
            "domain_name": domain_name,
            "dataloader": temp_dataloader
        }
        res.append(Munch(dic))
    
    return res


def shuffle_data_stream(dataStream):
    
    data_stream = copy.deepcopy(dataStream)
    num_batches = len(data_stream)
    batch_size = data_stream[0].dataloader.batch_size
       
    all_inputs = []
    all_labels = []
    for ele in data_stream:
        for x, y in ele.dataloader:
            all_inputs.append(x)
            all_labels.append(y)

    all_inputs = torch.cat(all_inputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    indices = np.array([i for i in range(num_batches*batch_size)])
    np.random.shuffle(indices)
    
    all_inputs = all_inputs[indices]
    all_labels = all_labels[indices]
    
    res = []
    
    for i in range(num_batches):
        temp_inputs = all_inputs[i*batch_size : (i+1)*batch_size]
        temp_labels = all_labels[i*batch_size : (i+1)*batch_size]
        temp_dataset = TensorDataset(temp_inputs, temp_labels)
        temp_dataloader = DataLoader(temp_dataset, batch_size=batch_size, shuffle=False)    
        temp_dict = {"dataloader": temp_dataloader,
                     "domain_name": "random"}
        res.append(Munch(temp_dict))
        
    return res



"""
Given data and labels, seperate them into calibration data and validation data
"""
def get_cal_and_val(images, labels, n_per_class):
    
    # Group images by class
    class_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        class_indices[label.item()].append(idx)
    
    batch1_indices = []
    batch2_indices = []
    
    # Split each class
    for label, indices in class_indices.items():
        indices = torch.tensor(indices)
        perm = torch.randperm(len(indices))
        batch1_indices.extend(indices[perm[:n_per_class]].tolist())
        batch2_indices.extend(indices[perm[n_per_class:]].tolist())
    
    # Convert to tensors and shuffle
    batch1_indices = torch.tensor(batch1_indices)[torch.randperm(len(batch1_indices))]
    batch2_indices = torch.tensor(batch2_indices)[torch.randperm(len(batch2_indices))]
    
    # Create the batches
    batch1_images = images[batch1_indices]
    batch1_labels = labels[batch1_indices]
    
    batch2_images = images[batch2_indices]
    batch2_labels = labels[batch2_indices]
    
    cal_data = batch1_images
    cal_labels = batch1_labels
    val_data = batch2_images
    val_labels = batch2_labels

    return cal_data, cal_labels, val_data, val_labels



"""
Given a list of dataloaders, combine all the data togeher and form a new dataloader
"""
def combine_dataloaders(dataloaders_list):
    batch_size = dataloaders_list[0].batch_size
    datasets = [dataloader.dataset for dataloader in dataloaders_list]
    concatenated_dataset = ConcatDataset(datasets)
    combined_dataloader = DataLoader(concatenated_dataset, batch_size=batch_size, shuffle=False)
    return combined_dataloader


"""
"""
def collect_images_by_label(dataloader):
    # Initialize an empty dictionary to store batches by label
    label_to_images = {}
    
    # Iterate through the dataloader
    for images, labels in dataloader:
        # Convert labels to a list
        labels = labels.tolist()
        
        # Iterate over the images and their corresponding labels
        for img, lbl in zip(images, labels):
            if lbl not in label_to_images:
                label_to_images[lbl] = []  # Create new list for this label
            label_to_images[lbl].append(img)
    
    # Now, we need to prepare the final list of batches for each label
    images_by_label = []
    labels_list = []
    
    # For each label, append the batch of images and labels
    for lbl in sorted(label_to_images.keys()):  # Sort labels to maintain order
        images_by_label.append(torch.stack(label_to_images[lbl]))  # Stack the images
        labels_list.append(lbl)  # Collect label for each group
    
    labels_list = torch.tensor(labels_list)
    images_by_label = torch.stack(images_by_label)

    return images_by_label, labels_list
