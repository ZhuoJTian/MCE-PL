import gc
import pickle
import logging
import numpy as np
import torch.optim as optim
import torch
import torch.nn as nn
import copy
# from sparse_regu import sparse_regularization
from torch.utils.data import DataLoader

from sparse_regu import sparse_regularization 
from utils.model_att3 import (
    switch_to_prune,
    switch_to_finetune,
)

logger = logging.getLogger(__name__)


class Client(object):
    """Class for client object having its own (private) data and resources to train a model.

    Participating client has its own dataset which are usually non-IID compared to other clients.
    Each client only communicates with the center server with its trained parameters or globally aggregated parameters.

    Attributes:
        id: Integer indicating client's id.
        data: torch.utils.data.Dataset instance containing local data.
        device: Training machine indicator (e.g. "cpu", "cuda").
        __model: torch.nn instance as a local model.
    """
    def __init__(self, client_id, local_data_train, local_data_val, local_data_test, device):
        """Client object is initiated by the center server."""
        self.id = client_id
        self.data_train = local_data_train
        self.data_test = local_data_test
        self.data_val = local_data_val
        self.device = device
        self.lam = 0.001
        self.__model = None
        self.mask_local_old = 0
        self.mask_neig = 0
        self.ns_p = 0
        self.i_out = {}
        self.j_out = {}

    @property
    def model(self):
        """Local model getter for parameter aggregation."""
        return self.__model

    @model.setter
    def model(self, model):
        """Local model setter for passing globally aggregated model parameters."""
        self.__model = model

    def setup(self, **client_config):
        """Set up common configuration of each client; called by center server."""
        self.dataloader_train = DataLoader(self.data_train, batch_size=client_config["batch_size"], shuffle=True)
        self.dataloader_test = DataLoader(self.data_test, batch_size=client_config["test_batch_size"], shuffle=True)
        self.local_epoch = client_config["num_local_epochs"]
        self.criterion = client_config["criterion"]
        self.optimizer = client_config["optimizer"]
        self.mask_optimizer = client_config["mask_optimizer"]
        self.lr_policy = client_config["lr_policy"]
        self.mask_lr_policy = client_config["mask_lr_policy"]
        self.optimizer_list = (self.optimizer, self.mask_optimizer)
        self.regular = sparse_regularization(self.model, self.device)


    def client_update(self, ep, ini=False):
        """Update local model using local dataset."""
        self.model.to(self.device)
        self.model.train()
        for train_images, train_targets in DataLoader(self.data_train, batch_size=128, shuffle=True):
            switch_to_prune(self.model)
            self.mask_optimizer.zero_grad()
            loss_mask = self.criterion(self.model(train_images.to(self.device), ini,\
                                                [x.to(self.device) for x in self.mask_local_old],
                                                [x.to(self.device) for x in self.mask_neig]),\
                                                train_targets.to(self.device)) # + self.regular.group_lasso_regularization(self.lam)
            loss_mask.backward()
            self.mask_optimizer.step()
            break
        self.model.to("cpu")
        if self.device == "cuda": torch.cuda.empty_cache()
        

    def client_evaluate_vali(self, ini = False):
        """Evaluate local model using local dataset (same as training set for convenience)."""
        self.model.eval()
        self.model.to(self.device)
        test_loss, correct, test_loss_regu = 0, 0, 0
        with torch.no_grad():
            for data, labels in DataLoader(self.data_test, batch_size=len(self.data_test), shuffle=True):
                data, labels = data.float().to(self.device), labels.long().to(self.device)
                if ini==False:
                    outputs = self.model(data, ini,
                                    [x.to(self.device) for x in self.mask_local_old], 
                                    [x.to(self.device) for x in self.mask_neig])
                else:
                    outputs = self.model(data, ini, 0, 0)
                test_loss += self.criterion(outputs, labels).item()
                test_loss_regu += self.regular.group_lasso_regularization(self.lam)
                
                predicted = outputs.argmax(dim=1, keepdim=True)
                correct += predicted.eq(labels.view_as(predicted)).sum().item()

                if self.device == "cuda": torch.cuda.empty_cache()
        self.model.to("cpu")

        test_loss = test_loss 
        test_loss_regu = test_loss_regu
        test_accuracy = correct/len(self.data_test)
        return test_loss, test_loss_regu, test_accuracy

    def client_evaluate_train(self, ini = False):
        """Evaluate local model using local dataset (same as training set for convenience)."""
        self.model.eval()
        self.model.to(self.device)
        test_loss, correct, test_loss_regu = 0, 0, 0
        with torch.no_grad():
            for data, labels in DataLoader(self.data_train, batch_size=len(self.data_train), shuffle=True):
                data, labels = data.float().to(self.device), labels.long().to(self.device)
                if ini==False:
                    outputs = self.model(data, ini,
                                    [x.to(self.device) for x in self.mask_local_old], 
                                    [x.to(self.device) for x in self.mask_neig])
                else:
                    outputs = self.model(data, ini, 0, 0)
                test_loss += self.criterion(outputs, labels).item()
                test_loss_regu += self.regular.group_lasso_regularization(self.lam)

                predicted = outputs.argmax(dim=1, keepdim=True)
                correct += predicted.eq(labels.view_as(predicted)).sum().item()

                if self.device == "cuda": torch.cuda.empty_cache()
        self.model.to("cpu")

        test_loss = test_loss 
        test_loss_regu = test_loss_regu 
        test_accuracy = correct/len(self.data_train)
        return test_loss, test_loss_regu, test_accuracy

    def def_ns(self, mask_local_old, mask_neig):
        self.mask_local_old = mask_local_old
        self.mask_neig = mask_neig

    def get_grad_pop(self):
        grad_list = []
        for i, v in self.model.named_modules():
            if hasattr(v, "popup_scores_local"):
                if getattr(v, "popup_scores_local") is not None:
                    grad_list.append(getattr(v, "popup_scores_local").grad.detach())
        return grad_list
    
    def get_pop_key(self):
        model_dict = self.model.state_dict()
        list_com = ['popup_scores_local'] #   'layer2', ,'layer4'
        key_pop = [k for k in model_dict if any(kk in k for kk in list_com)]
        return key_pop

    def get_mask_list(self, ini):
        # return a list of local weight for conv layer
        mask_list = []
        self.model.eval()
        self.model.to(self.device)
        if ini==False:
            
            with torch.no_grad():
                for data, labels in self.dataloader_train:
                    data, labels = data.float().to(self.device), labels.long().to(self.device)
                    outputs = self.model(data, ini,
                                    [x.to(self.device) for x in self.mask_local_old], 
                                    [x.to(self.device) for x in self.mask_neig])
                    break
            self.model.to("cpu")
            if self.device == "cuda": torch.cuda.empty_cache()
            mask_list = []
            coef_list = []
            for (name, vec) in self.model.named_modules():
                if hasattr(vec, "mask"):
                    attr = getattr(vec, "mask")
                    if attr is not None:
                        mask_list.append(attr.data.to("cpu"))
        else:
            with torch.no_grad():
                for data, labels in self.dataloader_train:
                    data, labels = data.float().to(self.device), labels.long().to(self.device)
                    outputs = self.model(data, ini, 0, 0)
                    break
            self.model.to("cpu")
            if self.device == "cuda": torch.cuda.empty_cache()
            mask_list = []
            coef_list = []
            for (name, vec) in self.model.named_modules():
                if hasattr(vec, "mask"): #popup_scores_local
                    attr = getattr(vec, "mask")
                    if attr is not None:
                        mask_list.append(attr.data.to("cpu"))

        return mask_list, coef_list

    def client_savemodel(self, path):
        torch.save(self.model.state_dict(), path+str(self.id)+"_alex.pt")

