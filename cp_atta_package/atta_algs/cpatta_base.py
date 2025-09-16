from munch import Munch
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from abc import ABC, abstractmethod
import gc
import copy

from cp_atta_package.utils import initialize, model_utils, uncertainty_measure, data_utils, domain_shift_degree_eval
from cp_atta_package.utils.register import register


"""
TopN BottomM base method
"""
class CPATTA_Base(ABC):

    def __init__(self, config):

        self.oracle_num_per_batch = config.atta.oracle_num_per_batch
        self.pseudo_num_per_batch = config.atta.pseudo_num_per_batch
        self.device = config.device
        self.seed = config.seed
        self.train_bs = config.train.batch_size
        self.lr = config.train.lr
        
        # self.model: realtime model
        # self.teacher: pretrained model
        self.model =  register.model_load_funcs[ f"load_{config.dataset.name.lower()}_model" ](config).to(self.device)
        self.teacher = copy.deepcopy(self.model.to('cpu'))
        self.model.to(self.device)
        self.teacher.to(self.device)

        self.train_method = config.train.method
        self.cold_start = config.train.cold_start
        self.stop_tol = config.train.stop_tol
        self.pseudo_labeled_storage = None
        self.oracle_labeled_storage = None
        self.pseudo_indices = None
        self.oracle_indices = None

        initialize.set_random_seeds(self.seed)
        assert self.train_method in ["simple", "simatta", "separate"]
       
        # store loss and corresponding coefficients
        self.during_training_oracle_loss = 0
        self.during_training_pseudo_loss = 0
        self.pseudo_loss_coef = None
        self.oracle_loss_coef = None

        
        self.w1_ema = 0
        self.w2_ema = 0
        self.mo = 0.8




    def get_top_bottom_indices(self, arr, topN, bottomM):
        """
        Returns the indices of the top N elements and bottom M elements from a 1D NumPy array.

        Parameters:
        arr (numpy.ndarray): The input 1D array.
        N (int): The number of top elements to find.
        M (int): The number of bottom elements to find.

        Returns:
        tuple: A tuple containing two lists:
            - The indices of the top N elements.
            - The indices of the bottom M elements.
        """

        N = topN
        M = bottomM
        
        if N < 0 or M < 0:
            raise ValueError("N and M must be non-negative integers.")


        # The below snip of code does not handle ties
        """  
        # Get the indices of the sorted array
        sorted_indices = np.argsort(arr)

        # Top N elements (last N in sorted array)
        top_indices = sorted_indices[-N:] if N > 0 else []

        # Bottom M elements (first M in sorted array)
        bottom_indices = sorted_indices[:M] if M > 0 else []

        return top_indices, bottom_indices
        """

        # Convert the array to a numpy array for easy manipulation
        arr = np.array(arr)
        
        # Get the sorted unique values for the top and bottom
        sorted_unique_vals = np.unique(arr)[::-1]  # Sorted descending
        sorted_unique_bottom_vals = np.unique(arr)  # Sorted ascending
        
        # Initialize result lists
        top_indices = []
        bottom_indices = []
        
        # Find indices of top N elements
        top_count = 0
        for val in sorted_unique_vals:
            # Get all indices where the element matches the value
            current_indices = np.where(arr == val)[0].tolist()
            if top_count + len(current_indices) <= N:
                top_indices.extend(current_indices)
                top_count += len(current_indices)
            else:
                # Randomly select remaining indices if necessary
                remaining = N - top_count
                top_indices.extend(np.random.choice(current_indices, remaining, replace=False).tolist())
                top_count = N
                break
        
        # Find indices of bottom M elements
        bottom_count = 0
        for val in sorted_unique_bottom_vals:
            current_indices = np.where(arr == val)[0].tolist()
            if bottom_count + len(current_indices) <= M:
                bottom_indices.extend(current_indices)
                bottom_count += len(current_indices)
            else:
                remaining = M - bottom_count
                bottom_indices.extend(np.random.choice(current_indices, remaining, replace=False).tolist())
                bottom_count = M
                break

        return np.array(top_indices), np.array(bottom_indices)


    def update_storage(self, storage, data, feats, true_labels, pseudo_labels):
        if storage is None:
            storage = Munch()
            storage.data = data
            storage.true_labels = true_labels
            storage.pseudo_labels = pseudo_labels
            storage.feats = feats
            storage.num_elem = lambda: len(storage.data)
        else:
            storage.data = torch.cat([storage.data, data])
            storage.feats = torch.cat([storage.feats, feats])
            storage.true_labels = torch.cat([storage.true_labels, true_labels])
            storage.pseudo_labels = torch.cat([storage.pseudo_labels, pseudo_labels])
        return storage
    

    @torch.enable_grad()
    def simple_train(self):

        # initialize loss accumulator
        self.during_training_oracle_loss = 0
        self.during_training_pseudo_loss = 0
        
        self.model.train()

        data = torch.cat([self.oracle_labeled_storage.data, self.pseudo_labeled_storage.data], dim=0)
        labels = torch.cat([self.oracle_labeled_storage.true_labels, self.pseudo_labeled_storage.pseudo_labels], dim=0)
        tt_dataloader = DataLoader(TensorDataset(data, labels), batch_size=self.train_bs, shuffle=True)
        
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        
        # define is_in_batch lambda function
        is_in_batch = lambda batch_of_tensors, image_tensor: any(torch.equal(image_tensor, tensor) for tensor in batch_of_tensors)


        for idx, (x, y) in enumerate(tt_dataloader):
           
            # move data
            x = x.to(self.device)
            y = y.to(self.device)


            # inference
            outputs = self.model(x)
            

            # accumulate loss of two dypes of data
            individual_loss = F.cross_entropy(outputs, y, reduction='none')
            assert len(individual_loss) == x.shape[0]
            oracle_flags = torch.tensor([1 if is_in_batch(self.oracle_labeled_storage.data, datum.cpu()) else 0 for datum in x], dtype=torch.float32).to(self.device)
            pseudo_flags = torch.tensor([1 if is_in_batch(self.pseudo_labeled_storage.data, datum.cpu()) else 0 for datum in x], dtype=torch.float32).to(self.device)
            # print(f"oracle flag: {oracle_flags}")
            # print(f"pseudo flag: {pseudo_flags}")
            if oracle_flags.sum() != 0:
                self.during_training_oracle_loss += (torch.dot(individual_loss, oracle_flags) / oracle_flags.sum()).item()
            if pseudo_flags.sum() != 0:
                self.during_training_pseudo_loss += (torch.dot(individual_loss, pseudo_flags) / pseudo_flags.sum()).item()


            # individual loss reduction
            loss = individual_loss.mean()

    
            # Backward pass and optimize
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()


    
    @torch.enable_grad()
    def simatta_train(self):
        
        # initialize loss accumulator
        self.during_training_oracle_loss = 0
        self.during_training_pseudo_loss = 0

        self.model.train()

        alpha = self.oracle_labeled_storage.num_elem() / (self.oracle_labeled_storage.num_elem() + self.pseudo_labeled_storage.num_elem())
        if self.pseudo_labeled_storage.num_elem() < self.cold_start:
            alpha = min(0.2, alpha)

        self.pseudo_loss_coef = (1-alpha)
        self.oracle_loss_coef = alpha

        """
        pseudo labeled data are source liked data
        oracle labeled data are target liked data
        """

        source_loader = DataLoader(TensorDataset(self.pseudo_labeled_storage.data, self.pseudo_labeled_storage.pseudo_labels),
                                   batch_size=self.train_bs)
        
        target_loader = DataLoader(TensorDataset(self.oracle_labeled_storage.data, self.oracle_labeled_storage.true_labels),
                                   batch_size=self.train_bs)

        ST_loader = iter(zip(source_loader, target_loader))
        
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)


        loss_window = []
        tol = 0
        lowest_loss = float('inf')
        
        for i, ((S_data, S_targets), (T_data, T_targets)) in enumerate(ST_loader):
            S_data, S_targets = S_data.to(self.device), S_targets.to(self.device)
            T_data, T_targets = T_data.to(self.device), T_targets.to(self.device)

            L_S = F.cross_entropy(self.model(S_data), S_targets)
            L_T = F.cross_entropy(self.model(T_data), T_targets)

            # record loss
            self.during_training_oracle_loss += L_T.item()
            self.during_training_pseudo_loss += L_S.item()

            loss = (1 - alpha) * L_S + alpha * L_T
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if len(loss_window) < self.stop_tol:
                loss_window.append(L_T.item())
            else:
                mean_loss = np.mean(loss_window)
                tol += 1
                if mean_loss < lowest_loss:
                    lowest_loss = mean_loss
                    tol = 0
                if tol > self.stop_tol:
                    break
                loss_window = []


    @torch.enable_grad()
    def separate_train(self):

        # initialize loss accumulator
        self.during_training_oracle_loss = 0
        self.during_training_pseudo_loss = 0
        
        self.model.train()
        
        # train oracle data
        data = self.oracle_labeled_storage.data
        labels = self.oracle_labeled_storage.true_labels
        tt_dataloader = DataLoader(TensorDataset(data, labels), batch_size=self.train_bs, shuffle=True)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.oracle_lr, momentum=0.9)

        # optimizer = torch.optim.Adam(self.model.parameters(), lr=self.oracle_lr)
        # print(f"oracle optimizer name: {optimizer.__class__.__name__}")

        for idx, (x, y) in enumerate(tt_dataloader):
           
            # move data
            x = x.to(self.device)
            y = y.to(self.device)

            # inference
            outputs = self.model(x)
            
            # accumulate loss of two dypes of data
            individual_loss = F.cross_entropy(outputs, y, reduction='none')
            self.during_training_oracle_loss += individual_loss.sum().item()

            # individual loss reduction
            loss = individual_loss.mean()
    
            # Backward pass and optimize
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()



        # train pusedo data
        data = self.pseudo_labeled_storage.data
        labels = self.pseudo_labeled_storage.pseudo_labels
        tt_dataloader = DataLoader(TensorDataset(data, labels), batch_size=self.train_bs, shuffle=True)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.pseudo_lr, momentum=0.9)

        # optimizer = torch.optim.Adam(self.model.parameters(), lr=self.pseudo_lr)
        # print(f"pseudo optimizer name: {optimizer.__class__.__name__}")

        for idx, (x, y) in enumerate(tt_dataloader):
           
            # move data
            x = x.to(self.device)
            y = y.to(self.device)

            # inference
            outputs = self.model(x)
            
            # accumulate loss of two dypes of data
            individual_loss = F.cross_entropy(outputs, y, reduction='none')
            self.during_training_pseudo_loss += individual_loss.sum().item()

            # individual loss reduction
            loss = individual_loss.mean()
    
            # Backward pass and optimize
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()



    @torch.no_grad()
    def get_before_training_loss(self):
        
        self.model.eval()

        before_training_oracle_loss = 0
        before_training_pseudo_loss = 0
        oracle_number = 0
        pseudo_number = 0

        oracle_dataloader = DataLoader(TensorDataset(self.oracle_labeled_storage.data, self.oracle_labeled_storage.true_labels), batch_size=self.train_bs, shuffle=False)
        pseudo_dataloader = DataLoader(TensorDataset(self.pseudo_labeled_storage.data, self.pseudo_labeled_storage.pseudo_labels), batch_size=self.train_bs, shuffle=False)
        
        for data, target in oracle_dataloader:
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            before_training_oracle_loss += F.cross_entropy(output, target, reduction='sum').item()
            oracle_number += len(target)

        for data, target in pseudo_dataloader:
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            before_training_pseudo_loss += F.cross_entropy(output, target, reduction='sum').item()
            pseudo_number += len(target)

        return before_training_oracle_loss, before_training_pseudo_loss


    @torch.no_grad()
    def get_before_training_loss_given_samples(self, x, y):

        self.model.eval()

        total_loss = 0.0
        total_samples = 0
        data, target = x.to(self.device), y.to(self.device)

        with torch.no_grad():
            logits = self.model(data)
            loss = F.cross_entropy(logits, target, reduction='none')
            total_loss += loss.sum().item()
            total_samples += data.size(0)
        
        return total_loss / total_samples if total_samples > 0 else 0.0
    

    @abstractmethod
    def select_data(self, tt_dataloader, cal_dataloader=None):
        pass

    @abstractmethod
    def train_model(self):
        pass

    def destroy_model(self):
        del self.model
        torch.cuda.empty_cache()
        gc.collect()


    def configure_model(self):  
        # self.edge_model = self.model[1]
        # self.cloud_model = self.model[0]
        """Configure model for use with tent."""
        # train mode, because tent optimizes the model to minimize entropy
        self.model.eval()  # eval mode to avoid stochastic depth in swin. test-time normalization is still applied
        # self.model[0].eval()
        # disable grad, to (re-)enable only what tent updates
        self.model.requires_grad_(False)
        # self.model[0].requires_grad_(False)
        # configure norm for tent updates: enable grad + force batch statisics
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                # m.track_running_stats = True
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            elif isinstance(m, nn.BatchNorm1d):
                m.requires_grad_(True)
            elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                m.requires_grad_(True)


    
    
    def collect_params(self):
        """Collect all trainable parameters.
        Walk the model's modules and collect all parameters.
        Return the parameters and their names.
        Note: other choices of parameterization are possible!
        """
        params = []
        names = []
        for nm, m in self.model.named_modules():
            for np, p in m.named_parameters():
                if np in ['weight', 'bias'] and p.requires_grad:
                    params.append(p)
                    names.append(f"{nm}.{np}")
        return params, names



    @torch.enable_grad()
    def debias_train(self):
        
        print("running debias train")

        params, _ = self.collect_params()
        # optimizer = torch.optim.SGD(params, lr=self.lr, momentum=0.9)
        optimizer = torch.optim.Adam(params, lr=self.lr)

        data = self.oracle_labeled_storage.data
        labels = self.oracle_labeled_storage.true_labels
        tt_dataloader = DataLoader(TensorDataset(data, labels), batch_size=self.train_bs, shuffle=True)
        loss_oracle = None

        for idx, (x, y) in enumerate(tt_dataloader):
           
            # move data
            x = x.to(self.device)
            y = y.to(self.device)

            # inference
            outputs = self.model(x)
            
            # accumulate loss of two dypes of data
            individual_loss = F.cross_entropy(outputs, y, reduction='none')

            if loss_oracle is None:
                loss_oracle = individual_loss
            else:
                # concatenate individual loss with loss oracle
                loss_oracle = torch.cat((loss_oracle, individual_loss), dim=0)

        # calculate the mean of loss oracle
        loss_oracle = loss_oracle.mean() if loss_oracle is not None else 0


        # train pusedo data
        data = self.pseudo_labeled_storage.data
        labels = self.pseudo_labeled_storage.pseudo_labels
        tt_dataloader = DataLoader(TensorDataset(data, labels), batch_size=self.train_bs, shuffle=True)
        loss_pseudo = None

        for idx, (x, y) in enumerate(tt_dataloader):
           
            # move data
            x = x.to(self.device)
            y = y.to(self.device)

            # inference
            outputs = self.model(x)
            
            # accumulate loss of two dypes of data
            individual_loss = F.cross_entropy(outputs, y, reduction='none')
            
            if loss_pseudo is None:
                loss_pseudo = individual_loss
            else:
                loss_pseudo = torch.cat((loss_pseudo, individual_loss), dim=0)
        
        loss_pseudo = loss_pseudo.mean() if loss_pseudo is not None else 0


        # Calculate gradients of loss1
        grad1 = torch.autograd.grad(loss_pseudo, list(param for param in self.model.parameters() if param.requires_grad), retain_graph=True)
        grad1_norm = torch.norm(torch.stack([g.norm() for g in grad1]))

        # Calculate gradients of loss2
        grad2 = torch.autograd.grad(loss_oracle, list(param for param in self.model.parameters() if param.requires_grad), retain_graph=True)
        grad2_norm = torch.norm(torch.stack([g.norm() for g in grad2]))
        
        w1 = 2 * grad2_norm / (grad2_norm + grad1_norm)
        w2 = 2 * grad1_norm / (grad1_norm + grad2_norm)
        w1 = w1.detach().item()
        w2 = w2.detach().item()
        

        self.w1_ema, self.w2_ema = self.update_w1_w2(w1, w2, self.w1_ema, self.w2_ema, self.mo)
        
        loss = self.w1_ema * loss_pseudo + self.w2_ema * loss_oracle

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


    @torch.no_grad()
    def update_w1_w2(self, w1, w2, w1_ema, w2_ema, momentum=0.8):
        if w1_ema == 0 and w2_ema == 0:
            w1_ema = w1
            w2_ema = w2
            return w1_ema, w2_ema
        else:
            w1_ema = momentum * w1_ema + (1-momentum) * w1
            w2_ema = momentum * w2_ema + (1-momentum) * w2
            return w1_ema, w2_ema