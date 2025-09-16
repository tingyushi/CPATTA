# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
import os
import numpy as np
from munch import Munch
import time
import random
from collections import defaultdict


from cp_atta_package.utils.register import register
from cp_atta_package.utils import initialize
from cp_atta_package.utils import data_utils


class _InfiniteSampler(torch.utils.data.Sampler):
    """Wraps another Sampler to yield an infinite stream."""
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            for batch in self.sampler:
                yield batch


class ActualSequentialSampler(torch.utils.data.Sampler):
    r"""Samples elements sequentially, always in the same order.
    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(self.data_source)

    def __len__(self):
        return len(self.data_source)


class FastDataLoader:
    """DataLoader wrapper with slightly improved speed by not respawning worker
    processes at every epoch."""
    def __init__(self, dataset, weights, batch_size, num_workers, sequential=False, subset=None):
        super().__init__()
        self.dataset = dataset
        if weights is not None:
            sampler = torch.utils.data.WeightedRandomSampler(weights,
                replacement=False,
                num_samples=batch_size)
        elif sequential:
            if subset is None:
                sampler = torch.utils.data.SequentialSampler(dataset)
            else:
                sampler = ActualSequentialSampler(subset)
        elif subset is not None:
            sampler = torch.utils.data.SubsetRandomSampler(subset)
        else:
            sampler = torch.utils.data.RandomSampler(dataset,
                replacement=False)
        self.sampler = sampler

        batch_sampler = torch.utils.data.BatchSampler(
            sampler,
            batch_size=batch_size,
            drop_last=False
        )

        self._infinite_iterator = iter(torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=_InfiniteSampler(batch_sampler),
            pin_memory=False,
            persistent_workers=True if num_workers > 0 else False
        ))

        self._length = len(batch_sampler)

    def __iter__(self):
        for _ in range(len(self)):
            yield next(self._infinite_iterator)

    def __len__(self):
        return self._length



class InfiniteDataLoader:
    def __init__(self, dataset, weights, batch_size, num_workers, sequential=False, subset=None):
        super().__init__()
        self.dataset = dataset
        if weights is not None:
            sampler = torch.utils.data.WeightedRandomSampler(weights,
                replacement=True,
                num_samples=batch_size)
        elif sequential:
            if subset is None:
                sampler = torch.utils.data.SequentialSampler(dataset)
            else:
                sampler = ActualSequentialSampler(subset)
        elif subset is not None:
            sampler = torch.utils.data.SubsetRandomSampler(subset)
        else:
            sampler = torch.utils.data.RandomSampler(dataset,
                replacement=True)
        self.sampler = sampler

        if weights == None:
            weights = torch.ones(len(dataset))

        batch_sampler = torch.utils.data.BatchSampler(
            sampler,
            batch_size=batch_size,
            drop_last=False)

        self._infinite_iterator = iter(torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=_InfiniteSampler(batch_sampler),
            pin_memory=False,
            persistent_workers=True if num_workers > 0 else False
        ))

    def __iter__(self):
        while True:
            yield next(self._infinite_iterator)

    def __len__(self):
        raise ValueError




def split_dataloader_by_label(dataloader, n_per_class, batch_size_, seed):
    dataset = dataloader.dataset
    targets = dataset.targets if hasattr(dataset, 'targets') else [label for _, label in dataset]

    # Group indices by class
    label_to_indices = defaultdict(list)
    for idx, label in enumerate(targets):
        label_to_indices[label].append(idx)

    # Set seed for reproducibility
    initialize.set_random_seeds(seed)

    selected_indices = []
    remaining_indices = []

    for label in sorted(label_to_indices.keys()):  # sort to keep ascending label order
        indices = label_to_indices[label]
        indices = random.sample(indices, len(indices))  # shuffle within class
        selected = indices[:n_per_class]
        remaining = indices[n_per_class:]

        selected_indices.extend(selected)
        remaining_indices.extend(remaining)

    # Sort selected_indices by label for ascending order
    selected_indices.sort(key=lambda idx: targets[idx])

    selected_subset = Subset(dataset, selected_indices)
    remaining_subset = Subset(dataset, remaining_indices)

    selected_loader = DataLoader(selected_subset, batch_size=batch_size_, shuffle=False)
    remaining_loader = DataLoader(remaining_subset, batch_size=batch_size_, shuffle=True)

    return selected_loader, remaining_loader





@register.data_load_func_register
def load_tinyimagenetc_data(config):
    
    severity = 5

    print(f"\n********** Loading {config.dataset.name.upper()} Dataset -- Severity: {severity} **********\n")

    tik_all = time.time()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])])

    domains = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
               'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
               'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
    
    print(f"Domains: {domains}")

    test_envs = [10] + [i for i in range(15) if i != 10]

    tik = time.time()
    
    initialize.set_random_seeds(config.seed)
    image_folders = []
    for idx, domain in enumerate(domains):
        print(f"Loading {domain} image folder: {idx+1}/{len(domains)}")
        path = os.path.join(config.dataset.path, "Tiny-ImageNet-C", domain, str(severity))
        image_folders.append(ImageFolder(path, transform=transform))
    
    tok = time.time()
    print(f"Image folders loaded in {tok - tik:.2f} seconds") 




    tik = time.time()

    # generate domainwise data stream
    initialize.set_random_seeds(config.seed)
    fast_random = [np.random.permutation(len(env)) for env in image_folders]
    domainwise_dataloaders = []

    for idx, test_env in enumerate(test_envs):
        
        print(f"Loading {domains[test_env]} faster dataloader: {idx+1}/{len(test_envs)}")

        faster_loader = FastDataLoader(image_folders[test_env], 
                                       weights=None,
                                       batch_size=config.dataset.batch_size,
                                       num_workers=config.dataset.num_workers,
                                       subset=fast_random[test_env], sequential=True)
        domain_name = domains[test_env]
        domainwise_dataloaders.append(Munch({"domain_name": domain_name, "dataloader": faster_loader}))

    tok = time.time()
    print(f"Domainwise dataloaders generated in {tok - tik:.2f} seconds")

    # get calibration data and validation data
    tik = time.time()
    
    cal_dataloader, val_dataloader = split_dataloader_by_label(domainwise_dataloaders[0].dataloader, 
                                                               n_per_class=config.dataset.num_of_cal_data_per_class,
                                                               batch_size_=config.dataset.batch_size,
                                                               seed=config.seed)

    tok = time.time()
    print(f"Calibration data prepared in {tok - tik:.2f} seconds")

    tok_all = time.time()
    print(f"All data prepared in {tok_all - tik_all:.2f} seconds")

    return domainwise_dataloaders, cal_dataloader, val_dataloader