import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from tqdm import tqdm

from utils.eval import eval_net
from UNet import UNet

from torch.utils.data import DataLoader, random_split
from utils.dataloader import get_loader,test_dataset

train_img_dir = '../DS-TransUNet/data/Kvasir_SEG_Training_880/image/'
train_mask_dir = '../DS-TransUNet/data/Kvasir_SEG_Training_880/mask/'
val_img_dir = '../DS-TransUNet/data/Kvasir_SEG_Validation_120/images/'
val_mask_dir = '../DS-TransUNet/data/Kvasir_SEG_Validation_120/masks/'
dir_checkpoint = './checkpoints/'


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def cal(loader):
    tot = 0
    for batch in loader:
        imgs, _ = batch
        tot += imgs.shape[0]
    return tot


# 这个损失函数尝试同时优化预测的准确性（通过 WBCE）和预测边界的精确性（通过 WIoU）。通过给边缘更多的权重，鼓励模型在分割时更关注于对象的边缘部分。
def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask) # 用于强调mask中边缘部分的重要性
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay


def train_net(net, device, epochs=500, batch_size=1, lr=0.01, save_cp=True, n_class=1, img_size=512):
    train_loader = get_loader(train_img_dir, train_mask_dir, batchsize=batch_size, trainsize=img_size, augmentation = False)
    val_loader = get_loader(val_img_dir, val_mask_dir, batchsize=1, trainsize=img_size, augmentation = False)

    n_train = cal(train_loader)
    n_val = cal(val_loader)
    logger = get_logger('kvasir.log')

    logger.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Vailding size:   {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images size:     {img_size}
    ''')

    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs//5, lr/10) # 周期性调整学习率，以优化训练效果
    if n_class > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss() # 应用于二分类问题的损失函数，内部包含了sigmiod激活


    best_dice = 0 # 用于记录最好的dice系数
    size_rates = [384, 512, 640] # 三种不同shape的图像进行训练，默认图像大小是512×512
    for epoch in range(epochs): # 每个epoch中要处理n_train的训练数据，每个训练数据要转成3种不同的size分别处理一遍
        net.train() # 将模型转为training模式

        epoch_loss = 0
        b_cp = False
        Batch = len(train_loader) # 训练数据大小
        with tqdm(total=n_train*len(size_rates), ncols=100, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                for rate in size_rates: # 每个shape都训练一遍
                    imgs, true_masks = batch
                    trainsize = rate
                    if rate != 512: # train_loader中已经将图像的大小改成了512×512
                        imgs = F.upsample(imgs, size=(trainsize, trainsize), mode='bilinear', align_corners=True) # 将图像调整到特定的大小，使用双线性插值法
                        true_masks = F.upsample(true_masks, size=(trainsize, trainsize), mode='bilinear', align_corners=True) # 将mask调整到特定的大小，使用双线性插值法


                    imgs = imgs.to(device=device, dtype=torch.float32)
                    mask_type = torch.float32 if n_class == 1 else torch.long
                    true_masks = true_masks.to(device=device, dtype=mask_type)

                    masks_pred, l2, l3 = net(imgs) # 模型输出
                    loss1 = structure_loss(masks_pred, true_masks)
                    loss2 = structure_loss(l2, true_masks)
                    loss3 = structure_loss(l3, true_masks)
                    loss = 0.6*loss1 + 0.2*loss2 + 0.2*loss3 # 加权求和
                    epoch_loss += loss.item() # 每个epoch中将loss值累加起来

                    pbar.set_postfix(**{'loss (batch)': loss.item()})

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_value_(net.parameters(), 0.1) # 对模型梯度进行剪裁，防止梯度爆炸问题（防止出现过大的梯度使模型权重在一次迭代中更新过多）
                    optimizer.step()

                    pbar.update(imgs.shape[0])

        scheduler.step() # 调整模型的学习率
        val_dice = eval_net(net, val_loader, device)
        if val_dice > best_dice:
           best_dice = val_dice
           b_cp = True
        epoch_loss = epoch_loss / Batch
        logger.info('epoch: {} train_loss: {:.3f} epoch_dice: {:.3f}, best_dice: {:.3f}'.format(epoch + 1, epoch_loss, val_dice* 100, best_dice * 100))

        if save_cp and b_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + 'epoch:{}_dice:{:.3f}.pth'.format(epoch + 1, val_dice*100))
            logging.info(f'Checkpoint {epoch + 1} saved !')



def get_args():
    parser = argparse.ArgumentParser(description='Train the model on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=1000,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=None,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--img_size', dest='size', type=int, default=512,
                        help='The size of the images')
    parser.add_argument('--optimizer', type=str,
                        default='Adam', help='choosing optimizer Adam or SGD')
    parser.add_argument('--decay_rate', type=float,
                        default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int,
                        default=50, help='every n epochs decay learning rate')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net = UNet(128, 1)
    net = nn.DataParallel(net, device_ids=[0]) # 启动分布式计算，将数据按batch平分到每块GPU上，将模型在每个GPU上都复制一份，专用于处理相应部分的数据
    net = net.to(device)

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
    logging.info(f'Model loaded from {args.load}')

    try:
        train_net(net=net, epochs=args.epochs, batch_size=args.batchsize, lr=args.lr, device=device, img_size=args.size)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
