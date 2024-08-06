import os
from time import time

import argparse
import torch
import math
import numpy as np
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import datetime
import scipy.io as sio
from model import HCVTNet
from model import CONFIGS
from data_loader import SeisDataset
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import random

if __name__ == '__main__':
    # train parameter
    parser = argparse.ArgumentParser(description='3D Reconstruction Training')
    parser.add_argument('--gt_path', type=str, default='./dataset/seg45_new/test/gt/data.mat')
    parser.add_argument('--mask_path', type=str, default='./dataset/seg45_new/test/mask_cm20/data.mat')
    parser.add_argument('--mask_test_path', type=str, default='./dataset/kerry/test/112-64-224/mask_im80/pseudo.mat')
    parser.add_argument('--result_path', type=str, default='results',
                        help='training output path that save logs and ckpt')
    parser.add_argument('--epoch', type=int, default=30000,
                        help='the number of epoch')
    parser.add_argument('--leaing_rate', type=float, default=1e-4,
                        help='the value of leaing_rate')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='the value of weight_decay')
    parser.add_argument('--optim', type=str, default='adam',
                        help='network optimizer')
    parser.add_argument('--stage', type=int, default=1,
                        help='the number of stage of reconstruction network')
    parser.add_argument('--last_stage_path', type=str, default=None,
                        help='the path where the output of the previous/last stage of the network is saved')

    # model parameter
    parser.add_argument("--loss", type=str, default='MSE',
                        help='loss function')
    parser.add_argument("--check_point_save", type=int, default=1000,
                        help='epoch interval to save the model.pth')
    parser.add_argument("--pretrain_model_path", type=str, default=None,
                        help='path of pretrain model')

    args = parser.parse_args()

    current_time = datetime.datetime.now()

    # parameter
    cudnn.benchmark = True
    Epoch = args.epoch
    leaing_rate = args.leaing_rate
    batch_size = 1
    pin_memory = False
    stage = args.stage
    result_lists = []
    device = "cuda:0"

    def miss_pseudo_trace(miss_position, miss_type='IM', missing_coefficient=0.35):
        pseudo_miss_position = np.copy(miss_position.cpu())
        shape = pseudo_miss_position.shape

        if miss_type == "IM":
            pseudo_miss_position = np.reshape(pseudo_miss_position, [shape[0], shape[1], shape[2], shape[3] * shape[4]])
            indices = np.squeeze(np.argwhere(pseudo_miss_position[0, 0, 0, :] == 1))
            all_len = indices.shape[0]
            # b c t x y
            miss_id = np.array(random.sample(list(indices), np.int32(np.around(all_len * missing_coefficient))))

            for xy in miss_id:
                pseudo_miss_position[:, :, :, xy] = 0

            missed_position = pseudo_miss_position.reshape((shape[0], shape[1], shape[2], shape[3], shape[4]))
        return missed_position

    def cal_mse_loss(x, label, mask):
        diff = x - label
        squared_diff = diff ** 2
        weighted_squared_diff = squared_diff * mask
        cnt_nonzero = torch.count_nonzero(mask).float()
        loss = torch.sum(weighted_squared_diff) / cnt_nonzero
        return loss

    def mse_loss(x, label):
        diff = x - label
        squared_diff = diff ** 2
        loss = torch.sum(squared_diff)
        return loss

    def l1_loss(x, label):
        return torch.sum(torch.abs(x - label))

    # Network
    config_vit = CONFIGS['HCVT-Net']
    net = HCVTNet(config_vit, img_size=(128, 128, 32))
    net = net.to(device)

    if args.pretrain_model_path is not None:
        net.load_state_dict(torch.load(args.pretrain_model_path))

    # Dataloader
    if stage == 1:
        train_dataset = SeisDataset( args.gt_path, args.mask_path, args.mask_test_path)
    else:
        train_dataset = SeisDataset(args.gt_path, args.mask_path, args.mask_test_path)
    train_dl = DataLoader(train_dataset, batch_size)

    # optimizer
    if args.optim == 'adam':
        opt = torch.optim.Adam(net.parameters(), lr=leaing_rate, weight_decay=args.weight_decay)
    else:
        assert False, print('Not implemented optimizer: {}'.format(args.optim))

    # log
    result_path = args.result_path
    dir_name = "seg45_new"
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(os.path.join(result_path, 'logs')):
        os.mkdir(os.path.join(result_path, 'logs'))
    if not os.path.exists(os.path.join(result_path, dir_name)):
        os.mkdir(os.path.join(result_path, dir_name))
    log_dir = os.path.join(result_path, 'logs')
    ckpt_dir = os.path.join(result_path, dir_name)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)
    train_writer = SummaryWriter(os.path.join(log_dir, 'train'))
    start = time()
    cptn = args.check_point_save

    # Training loop
    for epoch in range(Epoch):
        mean_loss = []

        for step, (miss_position, miss_data, test_position, test_data, gt_data) in enumerate(train_dl):
            miss_position = miss_position.to(device)
            miss_data = miss_data.to(device)

            # sample
            pseudo_miss_position = miss_pseudo_trace(miss_position, 'IM', 0.35)
            pseudo_miss_position = torch.from_numpy(pseudo_miss_position).to(device)

            gt_mask_pseudo = torch.multiply(miss_data, pseudo_miss_position)

            is_reverse_polarity = random.randint(0, 1)
            if is_reverse_polarity:
                gt_mask_pseudo = -gt_mask_pseudo

            outputs = net(gt_mask_pseudo)

            if is_reverse_polarity:
                outputs = -outputs

            loss = cal_mse_loss(outputs, miss_data, miss_position - pseudo_miss_position)

            mean_loss.append(loss)

            opt.zero_grad()
            loss.backward()
            opt.step()

            if epoch % 1 == 0:
                print('epoch:{}, step:{}, loss:{:.6f}, time:{:.3f} min'
                    .format(epoch, step, loss.item(), (time() - start) / 60))
            train_writer.add_scalar('TrainBatchLoss', loss.item(), epoch)

            if (epoch % cptn == 0 and epoch != 0) or (epoch == Epoch - 1):
                if not os.path.exists(ckpt_dir):
                    os.makedirs(ckpt_dir)

                predict = np.squeeze(np.array(outputs.cpu().detach().numpy(), dtype='float32'))
                gt_data = np.squeeze(np.array(gt_data.cpu().detach().numpy(), dtype='float32'))
                ssim = structural_similarity(gt_data, predict, data_range=2)
                psnr = peak_signal_noise_ratio(gt_data, predict, data_range=2)
                print(ssim, psnr)
                result_lists.append(
                    {
                        'epoch': epoch,
                        'ssim': ssim,
                        'psnr': psnr
                    }
                )
                sio.savemat(os.path.join(ckpt_dir, str(epoch) + '-ssim-' + str(ssim) + '-psnr-' + str(psnr) + '_.mat'),
                            {'data': np.squeeze(outputs.detach().cpu().numpy())})

                mean_loss = sum(mean_loss) / len(mean_loss)
                train_writer.add_scalar('TrainEpochLoss', mean_loss, epoch)
                torch.save(net.state_dict(), ckpt_dir + '/UNet{}-{:.3f}-{:.3f}.pth'.format(epoch, loss.item(), mean_loss))
        if epoch % 1000 == 0:
            print(f"{epoch} learning rate：{opt.param_groups[0]['lr']:.6f}")

    result_lists.sort(key=lambda x: x["psnr"], reverse=False)
    best_psnr_item = result_lists[-1]
    print("")
    print("")
    print("")
    print("")
    print("")
    print("")
    print("********** training end **********")
    print(f"best psnr epoch is {best_psnr_item['epoch']}: ------ ssim: {best_psnr_item['ssim']} ------ psnr: {best_psnr_item['psnr']}")
    finish_time = datetime.datetime.now()

    print(current_time, finish_time)
    print(finish_time - current_time)
    print("*****************************")


