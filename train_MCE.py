from __future__ import absolute_import
from __future__ import print_function

import importlib
import logging
import os
import time
import random
import math
from pathlib import Path
from client_MCE import Client
import torch
import torch.nn as nn
from itertools import product

from Sample_parti_noiid3 import getAttr
from datasets import cifar10
from tqdm import tqdm
import copy
import numpy as np
from torch.distributions.bernoulli import Bernoulli
# import datasets_old
from models.alex_cifar_att2 import AlexNet
from args import parse_args
import torch.nn.functional as F

from utils.general_utils import (
    setup_seed,
    getModelSize_Mask
)
from utils.model_att3 import get_layers
from utils.eval import show_filters_alex
from utils.schedules import get_lr_policy, get_optimizer

cpu_num = 1
os.environ["OMP_NUM_THREADS"] = str(cpu_num)
os.environ["MKL_NUM_THREADS"] = str(cpu_num)

torch.cuda.set_device(5)
torch.autograd.set_detect_anomaly(True)
torch.set_num_threads(cpu_num)

device = "cuda:5" if torch.cuda.is_available() else "cpu"

def create_clients(local_datasets_train, local_datasets_val, local_datasets_test):
    clients = []
    for k in range(len(local_datasets_train)):
        client = Client(client_id=k, local_data_train=local_datasets_train[k],
                        local_data_val=local_datasets_val[k], 
                        local_data_test=local_datasets_test[k], 
                        device="cuda")
        clients.append(client)
    return clients

def set_up_clients(args, k_list):
    CCategoryToClients, LocalDist, LocaDis_test, SamplesToClients, SamplesToClients_test, MAX_NUM = getAttr()
    local_datasets_train, local_datasets_val, local_datasets_test= \
        cifar10.create_datasets(args.data_dir, args.dataset,
                                args.num_clients, LocalDist, LocaDis_test)
    print("load the datasets")
    # Ini_model = RestNet18_att()
    clients = create_clients(local_datasets_train, local_datasets_val, local_datasets_test)
    
    for client_id in range(args.num_clients):
        client = clients[client_id]
        ConvLayer, LinearLayer = get_layers(args.layer_type)
        model = AlexNet(
            ConvLayer, LinearLayer, num_classes=args.num_classes, k=k_list[client_id]).to(device)
        PATH = 'ini.pth'
        model.load_state_dict(torch.load(PATH),strict=False)
        client.model = copy.deepcopy(model).to("cpu")
        del(model)

        optimizer=get_optimizer(client.model, args)
        mask_optimizer=torch.optim.SGD(
                        client.model.parameters(),
                        lr=args.mask_lr,
                        momentum=args.momentum,
                        weight_decay=args.wd)
        lr_policy = get_lr_policy(args.lr_schedule)(optimizer, args)
        mask_lr_policy = get_lr_policy(args.mask_lr_schedule)(mask_optimizer, args)
        client.setup(batch_size=args.batch_size,
                     test_batch_size=args.test_batch_size,
                     criterion=nn.CrossEntropyLoss(),
                     num_local_epochs=1,
                     optimizer = optimizer,
                     mask_optimizer = mask_optimizer,
                     lr_policy = lr_policy,
                     mask_lr_policy = mask_lr_policy
                     )
    print("load the clients")
    return clients


def run_test(clients, num_clients, ep, path, cost, ini):
    train_acc = np.zeros(num_clients)
    train_reg = np.zeros(num_clients)
    train_loss = np.zeros(num_clients)
    val_acc = np.zeros(num_clients)
    val_reg = np.zeros(num_clients)
    val_loss = np.zeros(num_clients)
    for client_id in range(num_clients):
        t_loss, t_reg, t_acc = clients[client_id].client_evaluate_train(ini=ini)
        train_acc[client_id] = t_acc
        train_reg[client_id] = t_reg
        train_loss[client_id] = t_loss
        v_loss, v_reg, v_acc = clients[client_id].client_evaluate_vali(ini=ini)
        val_acc[client_id] = v_acc
        val_reg[client_id] = v_reg
        val_loss[client_id] = v_loss
        # print(client_id, val_acc[client_id])

    print(
        "Localtest: [epoch {}, {} train_inst, {} test_inst] Training ACC: {:.4f}, Training Reg: {:.4f}, Training_Loss: {:.4f}, "
        "Testing ACC: {:.4f}, Testing Reg: {:.4f}, Testing_Loss: {:.4f}, Cost: {:.4f}\n"
        .format(ep + 1, 60000, 10000, np.average(train_acc), np.average(train_reg), np.average(train_loss),
                np.average(val_acc), np.average(val_reg), np.average(val_loss), cost))
    
    with open(path + '/MCE_ind_3_lr2.txt', "a+") as f:
        f.write(
            "[epoch {}, {} train_inst, {} test_inst] Training ACC: {:.4f}, Training Reg: {:.4f}, Training Loss: {:.4f}, "
            "Testing ACC: {:.4f}, Testing Reg: {:.4f}, Testing Loss: {:.4f}, Cost: {:.4f}\n"
            .format(ep + 1, 60000, 10000, np.average(train_acc), np.average(train_reg), np.average(train_loss),
                    np.average(val_acc), np.average(val_reg), np.average(val_loss), cost))
    return np.average(val_acc)


def get_nsp(neig_location, mask_list):
    ns_p = []
    for k in range(len(mask_list[0])):
        ns = torch.stack([mask_list[i][k] for i in neig_location], dim=-1) #*weight_list[k]
        ns_p.append(ns)
    return ns_p


def get_new_masks(clients, mask_list, adj):
    for client_id in range(len(clients)):
        client = clients[client_id]
        mask_local = mask_list[client_id]
        neig_location = list(np.nonzero(adj[:, client_id])[0])
        mask_neig = get_nsp(neig_location, mask_list)
        client.def_ns(mask_local, mask_neig)
    return clients


def update_localpop(clients, Models, grad_list, adj, cost_a):
    cost_total = 0
    for client_id in range(len(clients)):
        cost = cost_a[client_id]
        Model = Models[client_id]
        Grad = grad_list[client_id]
        client = clients[client_id]
        neig_location = np.nonzero(adj[:, client_id])[0]
        client_vars_sum = copy.deepcopy(client.model.state_dict())
        num_neig = len(neig_location)
        key_pop = client.get_pop_key()
        for kk in range(len(key_pop)):
            k = list(key_pop)[kk]
            local_mask = client.mask_local_old[kk]
            neig_mask = client.mask_neig[kk]
            local_mask_t = copy.deepcopy(local_mask)
            neig_mask_t = copy.deepcopy(neig_mask)
            local_mask_t[local_mask_t==0] = -1
            neig_mask_t[neig_mask_t==0] = -1
            coefs = 1.0/num_neig * torch.ones((num_neig, 1))
            momen =  Grad[kk] * torch.matmul(neig_mask, coefs).squeeze(-1)
            client_vars_sum[k] = Model[k] - 10.0*momen
        client.model.load_state_dict(client_vars_sum)
        cost_total += cost*num_neig
    return clients, cost_total


def main():
    args = parse_args()

    # create resutls dir (for logs, checkpoints, etc.)
    result_main_dir = os.path.join(Path(args.result_dir), args.exp_name)

    os.makedirs(result_main_dir, exist_ok=True)
    setup_seed(args.seed)
    k_list = [0.3 for i in range(args.num_clients)]

    ConvLayer, LinearLayer = get_layers(args.layer_type)
    unstructured = True if args.layer_type == "unstructured" else False
    model = AlexNet(
        ConvLayer, LinearLayer, num_classes=args.num_classes).to(device)
    
    # set up the clients
    adj_matrix = np.loadtxt('20_adj_matrix.txt')
    clients = set_up_clients(args, k_list)

    client = clients[0]
    mask_shape_list = []
    Model = client.model.to("cpu").state_dict()
    all_keys = [j for j in list(Model.keys())]
    keys = [k for k in all_keys if ('popup_scores_local' in k)]
    for k in keys:
        mask_shape_list.append(Model[k].shape)

    mask_list = []
    for i in range(len(keys)):
        shape_mask = mask_shape_list[i]
        if i<=2:
            vars_ns = (Bernoulli(0.3).sample(shape_mask))
        else:
            vars_ns = (Bernoulli(0.9).sample(shape_mask))
        mask_list.append(vars_ns)

    # define the parameters in each node
    Models = [0] * args.num_clients
    for client_id in range(args.num_clients):
        client = clients[client_id]
        neig_location = list(np.nonzero(adj_matrix[:, client_id])[0])
        num_neig = len(neig_location)
        mask_local_old = mask_list
        mask_l_conv = mask_list[0:3]
        mask_l_linear = mask_list[3:5]
        mm_conv = [vars_np.unsqueeze(-1).repeat(1, 1, 1, 1, num_neig) for vars_np in mask_l_conv]
        mm_linear = [vars_np.unsqueeze(-1).repeat(1, 1, num_neig) for vars_np in mask_l_linear]
        mask_neig = copy.deepcopy(mm_conv+mm_linear)
        client.def_ns(mask_local_old, mask_neig)

    print("the masks are intialized")
    
    mask = []
    
    cost_epoch = []
    # Start training
    for epoch in range(args.epochs):
        mask_list = [0] * args.num_clients
        grad_list = [0] * args.num_clients
        cost_a = [0] * args.num_clients
        for client_id in tqdm(range(args.num_clients), ascii=True):
            client = clients[client_id]
            client.client_update(epoch, ini=False)

            mask_local, _ = client.get_mask_list(ini=False)
            mask_list[client_id] = copy.deepcopy(mask_local)
            grad_list[client_id] = copy.deepcopy(client.get_grad_pop())
            cost_a[client_id] = getModelSize_Mask(mask_local)
            Models[client_id] = copy.deepcopy(client.model.state_dict())
        # evaluate on test set
        mask.append(mask_list)
        # define the local, neig and parameters for clients for next interation
        clients = get_new_masks(clients, mask_list, adj_matrix)
        clients, cost_total = update_localpop(clients, Models, grad_list, adj_matrix, cost_a)
        acc_val = run_test(clients, args.num_clients, epoch, str(result_main_dir), cost_total, ini=False)
        cost_epoch.append(cost_total)
        # print(cost_a)

if __name__ == "__main__":
    main()

