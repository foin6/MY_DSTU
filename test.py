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

from UNet import UNet

from torch.utils.data import DataLoader, random_split
from utils.dataloader import get_loader,test_dataset
from utils.eval import dice_coeff
from PIL import Image, ImageDraw, ImageFont

pred_path = 'path/to/the/dictionary/of/predict/results'
gt_path = 'path/to/the/dictionary/of/ground/truth'

def eval_net(net, loader, device, n_class=1):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval() # 启动模型的评估模式
    mask_type = torch.float32 if n_class == 1 else torch.long
    n_val = len(loader)
    pred_idx = 0
    gt_idx = 0
    img_idx = 0
    flag = False

    with tqdm(total=n_val, desc='Validation round', ncols=100, unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks = batch
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            mask_pred, _, _ = net(imgs)
            pred = torch.sigmoid(mask_pred)
            pred = (pred > 0.5).float() # 这样子拿到mask，设一个阈值
            for i in range(pred.shape[0]): # 按batch取出图像
                flag = False # dice系数未达要求
                img = pred[i]
                mask = true_masks[i]
                # calculate the dice coefficient
                total_dice, n = dice_coeff(pred, true_masks)
                dice = total_dice / n
                if dice < 0.95 or dice > 1.0:
                    pred_idx += 1
                    continue
                img = img.squeeze(0).cpu().numpy()
                img = Image.fromarray((img * 255).astype(np.uint8))
                # 右下角写入文本
                draw = ImageDraw.Draw(img)
                font = ImageFont.load_default()
                width, height = img.size
                text = "Dice: {:.3f}".format(dice.cpu())
                text_bbox = draw.textbbox((0, 0), text, font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                position = (width - text_width - 10, height - text_height - 10)
                draw.text(position, text, font=font, fill=255)
                # # 保存
                # print(img.size)
                # exit()
                img.save(pred_path+'/'+str(pred_idx)+'.png')
                pred_idx += 1
                
            for img in true_masks:
                img = img.squeeze(0).cpu().numpy()
                img = Image.fromarray((img * 255).astype(np.uint8))
                img.save(gt_path+'/'+str(gt_idx)+'.png')
                gt_idx += 1

            pbar.update()


def test_net(net,
              device,
              batch_size=1,
              n_class=1,
              img_size=512): # 图像大小是512×512


    val_img_dir = './dataset/BraTS/HGG_val_images/'
    val_mask_dir = './dataset/BraTS/HGG_val_masks/'

    val_loader = get_loader(val_img_dir, val_mask_dir, batchsize=batch_size, trainsize=img_size, shuffle=False, augmentation = False) # 这里实际上已经把images和mask的大小改成512×512了，并且不进行图像增强操作
    net.eval()

    eval_net(net, val_loader, device)


def get_args():
    parser = argparse.ArgumentParser(description='Train the model on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=4,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-f', '--load', dest='load', type=str, default=None,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--img_size', dest='size', type=int, default=384,
                        help='The size of the images')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net = UNet(128, 1)
    net = nn.DataParallel(net, device_ids=[0])
    net.to(device=device)

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device), False
        )
        logging.info(f'Model loaded from {args.load}')

    try:
        test_net(net=net,
                  batch_size=args.batchsize,
                  device=device,
                  img_size=args.size)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
