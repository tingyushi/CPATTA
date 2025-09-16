# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
import os
import numpy as np
from munch import Munch
import time
from os.path import join as opj

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



@register.data_load_func_register
def load_vlcs_data(config):

    print(f"\n********** Loading {config.dataset.name.upper()} Dataset **********\n")

    tik_all = time.time()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])])

    domains = ["C", "L", "S", "V"]

    print(f"Domains: {domains}")

    full_name_map = {"C": "Caltech101",
                     "L": "LabelMe",
                     "S": "SUN09",
                     "V": "VOC2007"}
    test_envs = [0, 1, 2, 3]

    tik = time.time()
    initialize.set_random_seeds(config.seed)
    image_folders = []
    for domain in domains:
        path = os.path.join( opj( config.dataset.path, config.dataset.name.upper() ) , full_name_map[domain])
        image_folders.append(ImageFolder(path, transform=transform))
    tok = time.time()
    print(f"Image folders loaded in {tok - tik:.2f} seconds")



    # generate domainwise data stream
    tik = time.time()
    initialize.set_random_seeds(config.seed)
    fast_random = [np.random.permutation(len(env)) for env in image_folders]
    fast_dataloaders = [FastDataLoader(env, 
                                       weights=None,
                                       batch_size=config.dataset.batch_size,
                                       num_workers=config.dataset.num_workers,
                                       subset=fast_random[i], sequential=True) for i, env in enumerate(image_folders)]
    
    domainwise_data_stream = []
    for test_env in test_envs:
        fast_loader = fast_dataloaders[test_env]
        domain_name = domains[test_env]
        for x, y in fast_loader:
            dataset = TensorDataset(x, y)
            dataloader = DataLoader(dataset, batch_size=config.dataset.batch_size, shuffle=False)
            domainwise_data_stream.append(Munch({"domain_name": domain_name, "dataloader": dataloader}))
    tok = time.time()
    print(f"Domainwise data stream generated in {tok - tik:.2f} seconds")

    # generate domainwise dataloaders
    tik = time.time()
    domainwise_dataloaders = []
    for test_env in test_envs:
        cur_domain = domains[test_env]
        dataloader = data_utils.combine_dataloaders([d.dataloader for d in domainwise_data_stream if d.domain_name == cur_domain])
        domainwise_dataloaders.append(Munch({"domain_name": cur_domain, "dataloader": dataloader}))
    tok = time.time()
    print(f"Domainwise dataloaders generated in {tok - tik:.2f} seconds")

    # get calibration data
    tik = time.time()
    initialize.set_random_seeds(config.seed)
    p_data = torch.tensor(data_utils.extract_data(domainwise_dataloaders[0].dataloader))
    p_labels = torch.tensor(data_utils.extract_labels(domainwise_dataloaders[0].dataloader))
    cal_data, cal_labels, val_data, val_labels = data_utils.get_cal_and_val(images=p_data, 
                                                                            labels=p_labels, 
                                                                            n_per_class=config.dataset.num_of_cal_data_per_class)
    cal_dataset = TensorDataset(cal_data, cal_labels)
    cal_dataloader = DataLoader(cal_dataset, batch_size=config.dataset.batch_size, shuffle=False)
    tok = time.time()
    print(f"Calibration data prepared in {tok - tik:.2f} seconds")

    data_streams = Munch({"domainwise": domainwise_data_stream})

    tok_all = time.time()
    print(f"All data prepared in {tok_all - tik_all:.2f} seconds")

    return data_streams, domainwise_dataloaders, cal_dataloader 