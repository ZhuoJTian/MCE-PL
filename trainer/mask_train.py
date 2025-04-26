import time

import torch

from utils.eval import accuracy
from utils.general_utils import AverageMeter, ProgressMeter
from utils.model import (
    switch_to_bilevel,
    switch_to_prune,
    switch_to_finetune,
)

from sparse_regu import sparse_regularization 

def train(
        model, device, train_loader, criterion, optimizer_list, regu=True, ini=False
):
    # print("->->->->->->->->->-> One epoch with Natural training <-<-<-<-<-<-<-<-<-<-")
    train_loader, val_loader = train_loader
    
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    top1 = AverageMeter("Acc_1", ":6.2f")
    top5 = AverageMeter("Acc_5", ":6.2f")

    optimizer, mask_optimizer = optimizer_list
    lr2 = 0.001
    print_freq = 10
    model.train()
    end = time.time()

    regular = sparse_regularization(model, device)

    for i, (train_data_batch, val_data_batch) in enumerate(zip(train_loader, val_loader)):
        train_images, train_targets = train_data_batch[0].to(device), train_data_batch[1].to(device)
        val_images, val_targets = val_data_batch[0].to(device), val_data_batch[1].to(device)
        if regu:
            switch_to_prune(model)
            mask_optimizer.zero_grad()
            loss_mask = criterion(model(train_images, ini), train_targets) # + regular.group_lasso_regularization(0.001)

            loss_mask.backward()
            mask_optimizer.step()

        else:
            switch_to_prune(model)
            mask_optimizer.zero_grad()
            loss_mask = criterion(model(train_images, ini), train_targets)
            loss_mask.backward()
            mask_optimizer.step()
    
        batch_time.update(time.time() - end)
        end = time.time()
