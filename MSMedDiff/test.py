import argparse
import os
import time
import random
from get_evaluation import get_result_to_txt
from read_nii import BRATSDataset
import cv2
import einops
import numpy
import numpy as np
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch import nn
from torch.utils.data import Dataset

from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets

from modules import UNet_conditional
# from new_modules import UNet_conditional
# from train_test import train_one_epoch, t

from train_sample import Diffusion, train

from multi_train_utils.distributed_utils import init_distributed_mode, dist, cleanup
from utils import keep_image_size_open, keep_image_size_open_rgb, save_images_double, save_images, \
    save_images_single_channel, save_images_three

def set_seed(SEED):
    # initialize random seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


# 设置随机种子
#set_seed(1234)
transform_image = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.Normalize((0.5,), (0.5,))  # 归一化
])

transform_seg_image = transforms.Compose([
    transforms.ToTensor(),  #
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.Normalize((0.5,), (0.5,))  # 归一化
])


# class Get_tager_sample(Dataset):
#     def __init__(self, path):
#         self.path = path
#         self.name = os.listdir(os.path.join(path, 'masks'))
#
#     def __len__(self):
#         return len(self.name)
#
#     def __getitem__(self, index):
#         segment_name = self.name[index]  # xx.png
#         segment_path = os.path.join(self.path, 'masks', segment_name)
#         image_path = os.path.join(self.path, 'images', segment_name)
#         segment_image = keep_image_size_open(segment_path)
#         image = keep_image_size_open_rgb(image_path)
#         # 增加分割图的维度，拓展成c，w，h
#         segment_image = torch.Tensor(np.array(segment_image))
#         segment_image = torch.unsqueeze(segment_image, 0)
#         # print('-----------------------------------------------segment_image-----------------------------------------------')
#         # print(segment_image.shape)
#         image = torch.Tensor(np.array(image))
#         image = einops.rearrange(image, "w h c ->  c w h")
#         # print('-----------------------------------------------image-----------------------------------------------')
#         # print(image.shape)
#         # return transform(image), torch.Tensor(np.array(segment_image))
#
#         return image, segment_image / 255


class Get_tager_sample(Dataset):
    def __init__(self, path):
        self.path = path
        # self.name = os.listdir(os.path.join(path, 'infection_mask'))
        self.name = os.listdir(os.path.join(path, 'flair'))

    def __len__(self):
        return len(self.name)

    def __getitem__(self, index):
        flair_name = self.name[index]  # xx.png
        segment_path = os.path.join(self.path, 'seg', flair_name)
        t1_path = os.path.join(self.path, 't1', flair_name)
        t1ce_path = os.path.join(self.path, 't1ce', flair_name)
        t2_path = os.path.join(self.path, 't2', flair_name)
        flair_path = os.path.join(self.path, 'flair', flair_name)
        segment_image = keep_image_size_open(segment_path)
        t1_image = keep_image_size_open_rgb(t1_path)
        t1_image = torch.Tensor(np.array(t1_image))
        t1_image = einops.rearrange(t1_image,"w h c -> c w h")
        t1ce_image = keep_image_size_open_rgb(t1ce_path)
        t1ce_image = torch.Tensor(np.array(t1ce_image))
        t1ce_image = einops.rearrange(t1ce_image, "w h c -> c w h")
        t2_image = keep_image_size_open_rgb(t2_path)
        t2_image = torch.Tensor(np.array(t2_image))
        t2_image = einops.rearrange(t2_image, "w h c -> c w h")
        flair_image = keep_image_size_open_rgb(flair_path)
        flair_image = torch.Tensor(np.array(flair_image))
        flair_image = einops.rearrange(flair_image, "w h c -> c w h")
        # 增加分割图的维度，拓展成c，w，h
        segment_image = torch.Tensor(np.array(segment_image))
        segment_image = torch.unsqueeze(segment_image, 0)
        image = np.stack([t1_image, t1ce_image, t2_image, flair_image], axis=0)
        image = einops.rearrange(image, 'b c w h -> (b c) w h')
        # print(type(image), 'image')
        # print( type(segment_image), 'segment_image')
        return image/ 255, segment_image / 255
        # return transform_image(image), transform_seg_image(segment_image)

def main(args):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training.")

    # 初始化各进程环境
    init_distributed_mode(args=args)

    rank = 0
    device = torch.device(args.device)
    print("设备", device)
    batch_size = args.batch_size
    # weights_path = args.weights
    #args.lr *= args.world_size  # 学习率要根据并行GPU的数量进行倍增
    checkpoint_path = ""
    dist.init_process_group('gloo', init_method='tcp://localhost:8081', world_size=1, rank=0)
    if rank == 0:  # 在第一个进程中打印信息，并实例化tensorboard
        print(args)
        print("initing over!")
        tb_write = SummaryWriter()
        print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')

    print(os.cpu_count())
    torch.cuda.empty_cache()
    # nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    # print('nw', nw)
    # if rank == 0:
    #     print('Using {} dataloader workers every process'.format(nw))
    print(torch.cuda.memory_allocated() / 1024 / 1024)
    # train_1 = Get_tager_sample(r"/media/ps/data/work/data/yb2yy/2023_lung_process")
    # test_1 = Get_tager_sample(r"/media/ps/data/work/data/yb2yy/2023_lung_process_test")

    test_1 = Get_tager_sample(r"/home/ps/test_code/data/brats_/brats/test")
    # train_1 = BRATSDataset("/home/yp/diskdata/data/MICCAI_BraTS2020_TrainingData", test_flag=False)

    # print('-----------------------------------------------test_1-----------------------------------------------')
    # print(type(test_1))
    # print(test_1.size())
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_1, shuffle=False)

    test_batch_sampler = torch.utils.data.BatchSampler(
        test_sampler, batch_size, drop_last=True)
    # nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    nw = 1  # number of workers
    if rank == 0:
        print('Using {} dataloader workers every process'.format(nw))

    test_loader = torch.utils.data.DataLoader(test_1,
                                              batch_sampler=test_batch_sampler,
                                              pin_memory=True,
                                              num_workers=nw,
                                              )

    # 单个时刻图像
    
    # 多个时刻图像
    # net = UNet_conditional(c_in=1, c_out=1, device=device, con_c_in=12).to(device)

    # checkpoint_path = "initial_weights.pt"
    # if rank == 0:
    #     torch.save(net.state_dict(), checkpoint_path)
    # dist.barrier()
    # net.load_state_dict(torch.load(checkpoint_path, map_location=device))

    path = "chest2_model"
    # checkpoint_path = "/media/ps/data/work/result/diff-base-chekpoint/chest/2200.pth"
    checkpoint_path_list = []
    checkpoint_list = os.listdir(path)
    # 按照文件名中的数字大小进行排序
    sorted_file_list = sorted(checkpoint_list, key=lambda x: int(x.split('.')[0]))

    #print(sorted_file_list)
    # print(checkpoint_list)
    for (index, checkpoint,) in enumerate(sorted_file_list):
        checkpoint_path = os.path.join(path, checkpoint)
        #print(checkpoint_path)
        checkpoint_path_list.append(checkpoint_path)
        # with open('checkpoint_path.txt', 'a') as file:
        #     file.write(f'checkpoint_path：{checkpoint_path}\n')
    i=-1
    for checkpoint_path in checkpoint_path_list[700:773:10]:
        checkpoint = os.path.basename(checkpoint_path)
        print(checkpoint_path)
        weight_dict = torch.load(checkpoint_path, map_location='cpu')
        # print('weight_dict:', weight_dict)
        # load_weight_dict = {k: v for k, v in weight_dict.items()
        #                     if net.state_dict()[k].numel() == v.numel()}
        # 单个时刻图像
        #net = UNet_conditional(c_in=1, c_out=1, device="cuda:0", con_c_in=3).to("cuda:0")
        
    
        # 多个时刻图像
        net = UNet_conditional(c_in=1, c_out=1, device=device, con_c_in=12).to(device)
        net.load_state_dict(weight_dict, strict=False)
        # if rank == 0:
        #     torch.save(net.state_dict(), checkpoint_path)
        dist.barrier()
        #net.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))

        # net = net.device(0)
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[0], find_unused_parameters=True)
        diffusion = Diffusion(img_size=256, device="cuda:0")  #, is_train=False
        # tb_write.add_scalar("mes", 1, 0)
        # net = torch.nn.parallel.DistributedDataParallel(net, device_ids= 0, find_unused_parameters=True)
        # for i, (images, lable) in enumerate(test_loader):
            
        #     img = images.to(device)
        #     lable = lable.to(device)
        i+=1
        img, lable = next(iter(test_loader))

        sampled_images = diffusion.sample_condition_decoder(net, n=img.shape[0], image=img, is_train=False)
        print(sampled_images.max())
        # numpy.save("1.npy",sampled_images.cpu().numpy())
        sampled_images_show = (sampled_images.clamp(-1, 1) + 1) / 2
        sampled_images_show = (sampled_images_show * 255).type(torch.uint8)
        save_images_double(sampled_images_show, os.path.join("test_result", "test_brats", f"{i}.jpg"),
                        lable)
        save_images(sampled_images, os.path.join("test_result", "test_brats_single", f"{0}.jpg"))
        save_images(lable, os.path.join("test_result", "test_brats_label", f"{0}.jpg"))
        save_images_single_channel(sampled_images,
                                os.path.join("test_result", "test_brats_image_single_channel"), 0)
        save_images_single_channel(lable, os.path.join("test_result", "test_brats_label_single_channel"),
                                0)
        get_result_to_txt(checkpoint)   

    # cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lrf', type=float, default=0.1)
    # 是否启用SyncBatchNorm
    parser.add_argument('--syncBN', type=bool, default=False)

    parser.add_argument('--is_load_net', type=bool, default=False)

    parser.add_argument('--save_weight', type=str, default='weight_att',
                        help='save weights path')

    parser.add_argument('--save_img', type=str, default='img_att',
                        help='save img path')

    parser.add_argument('--freeze-layers', type=bool, default=False)
    # 不要改该参数，系统会自动分配
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    # 开启的进程数(注意不是线程),不用设置该参数，会根据nproc_per_node自动设置
    parser.add_argument('--world-size', default=4, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', type=int, help='Specify the rank')
    parser.add_argument('--gpu', type=str, help='Specify the rank')
    opt = parser.parse_args()

    main(opt)
