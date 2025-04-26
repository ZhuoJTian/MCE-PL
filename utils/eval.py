import time

import torch
import torch.nn as nn
from tqdm import tqdm

from utils.general_utils import AverageMeter, ProgressMeter

import matplotlib.pyplot as plt

def data_normal(orign_data):
    abs_data = torch.abs(orign_data)
    d_min = orign_data.min()
    '''
    if d_min<0:
        orign_data += torch.abs(d_min)
        d_min = orign_data.min()'''
    d_max = orign_data.max()
    return abs_data, d_min, d_max


def get_output_for_batch(model, img, temp=1):
    """
        model(x) is expected to return logits (instead of softmax probas)
    """
    with torch.no_grad():
        out = nn.Softmax(dim=-1)(model(img) / temp)
        p, index = torch.max(out, dim=-1)
    return p.data.cpu().numpy(), index.data.cpu().numpy()


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def base(model, device, val_loader, criterion, args, epoch=0):
    """
        Evaluating on unmodified validation set inputs.
    """
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    top1 = AverageMeter("Acc_1", ":6.2f")
    top5 = AverageMeter("Acc_5", ":6.2f")
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, top1, top5], prefix="Test: "
    )

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, data in tqdm(enumerate(val_loader)):
            images, target = data[0].to(device), data[1].to(device)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % args.print_freq == 0:
                progress.display(i)

        progress.display(i)  # print final results

    return top1.avg, top5.avg


def show_filters_alex(model, mask, id, fig_path, conv_num='1'):
    localw1 = model.conv1.weight.detach().cpu().clone()
    local_w = localw1 * mask
    plt.figure(figsize=(30, 3))
    for i in range(local_w.shape[1]):
        for j in range(30):
            plt.subplot(local_w.shape[1], 30, i*30+j+1) 
            plt.axis('off')
            plt.imshow(local_w[j+34, i, :, :], cmap='gray', aspect='auto', vmin=-0.5, vmax=0.5)
    plt.savefig(fig_path+"id"+str(id)+"_conv"+str(conv_num))
    plt.close()

def show_filters_alex2(id, fig_path, weight_trans_list, conv_num='1'):
    local_w = weight_trans_list[0]
    plt.figure(figsize=(20, 1))
    for i in range(local_w.shape[1]):
        for j in range(local_w.shape[0]):
            plt.subplot(local_w.shape[1], local_w.shape[0], i*local_w.shape[0]+j+1) 
            plt.axis('off')
            plt.imshow(local_w[j, i, :, :], cmap='gray', aspect='auto', vmin=-0.5, vmax=0.5)
    plt.savefig(fig_path+"id"+str(id)+"_conv"+str(conv_num))
    plt.close()


def show_filters_res(model, id, fig_path, conv_num = 2):
    ll = list(model.children())
    if conv_num == 0:
        conv_need = list(model.children())[conv_num]
    else:
        conv_need = list(model.children())[conv_num][0].conv1
    localw1 = conv_need.w.cpu().clone()   

    # print("total of number of filter : ", localw.shape[0]*localw.shape[1])
    num = len(localw1)
    abs_data, d_min, d_max = data_normal(localw1)
    plt.figure(figsize=(10, 10))
    for i in range(localw1.shape[1]):
        for j in range(localw1.shape[0]):
            plt.subplot(localw1.shape[1], localw1.shape[0], i*localw1.shape[0]+j+1) 
            plt.axis('off')
            plt.imshow(localw1[j, i, :, :].detach(), cmap='gray', vmin=-0.01, vmax=0.01)
    plt.savefig(fig_path+"fig_id"+str(id)+"_conv"+str(conv_num))
    plt.close()

def show_filters_vgg1(model, id, fig_path, ep):
    conv_need = model.features[3]
    localw1 = conv_need.w.cpu().clone()   

    # print("total of number of filter : ", localw.shape[0]*localw.shape[1])
    plt.figure(figsize=(10, 10))
    for i in range(localw1.shape[1]):
        for j in range(localw1.shape[0]):
            plt.subplot(localw1.shape[1], localw1.shape[0], i*localw1.shape[0]+j+1) 
            plt.axis('off')
            plt.imshow(localw1[j, i, :, :].detach(), cmap='gray', vmin=-0.2, vmax=0.2)
    plt.savefig(fig_path+"fig_id"+str(id)+"ep"+str(ep))
    plt.close()

def show_filters_vgg2(model, id, fig_path, ep):
    conv_need = model.classifier[4]
    localw1 = conv_need.w.cpu().clone()   

    # print("total of number of filter : ", localw.shape[0]*localw.shape[1])
    plt.figure(figsize=(10, 10))
    plt.imshow(localw1.detach(), cmap='gray', vmin=-0.1, vmax=0.1)
    plt.savefig(fig_path+"2fig_id"+str(id)+"ep"+str(ep))
    plt.close()