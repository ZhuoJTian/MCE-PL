import time

import torch

from utils.eval import accuracy
from utils.general_utils import AverageMeter, ProgressMeter
from utils.model import (
    switch_to_bilevel,
    switch_to_prune,
    switch_to_finetune,
)


def train(
        model, device, train_loader, criterion, optimizer_list, ini=False):
    train_loader, val_loader = train_loader

    optimizer, mask_optimizer = optimizer_list

    model.train()
    end = time.time()

    for i, (train_data_batch, val_data_batch) in enumerate(zip(train_loader, val_loader)):
        train_images, train_targets = train_data_batch[0].to(device), train_data_batch[1].to(device)
        switch_to_finetune(model)
        output = model(train_images, ini)
        loss = criterion(output, train_targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
