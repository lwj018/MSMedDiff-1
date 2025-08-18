# main.py
import torch
import argparse
import torch.distributed as dist
import argparse
import os

import random

print(os.getcwd())
import einops
import numpy as np
import torch.optim as optim
from dice_loss.ema_pytorch import EMA
from torch.utils.data import Dataset, random_split

from torch.utils.tensorboard import SummaryWriter

from modules import UNet_conditional
from train_sample import Diffusion, train,val
from multi_train_utils.distributed_utils import cleanup
from utils import keep_image_size_open, keep_image_size_open_rgb, save_images_double, save_images, \
    save_images_single_channel, save_images_three
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


# 设置随机种子
#set_seed(42)


class Get_tager_sample(Dataset):
    def __init__(self, path):
        self.path = path
        self.name = os.listdir(os.path.join(path, 'masks'))

    def __len__(self):
        return len(self.name)

    def __getitem__(self, index):
        segment_name = self.name[index]  # xx.png
        segment_path = os.path.join(self.path, 'masks', segment_name)
        image_path = os.path.join(self.path, 'images', segment_name)
        segment_image = keep_image_size_open(segment_path)
        image = keep_image_size_open_rgb(image_path)
        # 增加分割图的维度，拓展成c，w，h
        segment_image = torch.Tensor(np.array(segment_image))
        segment_image = torch.unsqueeze(segment_image, 0)
        # print('-----------------------------------------------segment_image-----------------------------------------------')
        # print(segment_image.shape)
        image = torch.Tensor(np.array(image))
        image = einops.rearrange(image, "w h c ->  c w h")
        # print('-----------------------------------------------image-----------------------------------------------')
        # print(image.shape)
        # return transform(image), torch.Tensor(np.array(segment_image))

        return image/ 255, segment_image / 255

# class Get_tager_sample(Dataset):
#     def __init__(self, path):
#         self.path = path
#         # self.name = os.listdir(os.path.join(path, 'infection_mask'))
#         self.name = os.listdir(os.path.join(path, 'flair'))
#
#     def __len__(self):
#         return len(self.name)
#
#     def __getitem__(self, index):
#         flair_name = self.name[index]  # xx.png
#         segment_path = os.path.join(self.path, 'seg', flair_name)
#         t1_path = os.path.join(self.path, 't1', flair_name)
#         t1ce_path = os.path.join(self.path, 't1ce', flair_name)
#         t2_path = os.path.join(self.path, 't2', flair_name)
#         flair_path = os.path.join(self.path, 'flair', flair_name)
#         segment_image = keep_image_size_open(segment_path)
#         t1_image = keep_image_size_open_rgb(t1_path)
#         t1_image = torch.Tensor(np.array(t1_image))
#         t1_image = einops.rearrange(t1_image, "w h c -> c w h")
#         t1ce_image = keep_image_size_open_rgb(t1ce_path)
#         t1ce_image = torch.Tensor(np.array(t1ce_image))
#         t1ce_image = einops.rearrange(t1ce_image, "w h c -> c w h")
#         t2_image = keep_image_size_open_rgb(t2_path)
#         t2_image = torch.Tensor(np.array(t2_image))
#         t2_image = einops.rearrange(t2_image, "w h c -> c w h")
#         flair_image = keep_image_size_open_rgb(flair_path)
#         flair_image = torch.Tensor(np.array(flair_image))
#         flair_image = einops.rearrange(flair_image, "w h c -> c w h")
#         # 增加分割图的维度，拓展成c，w，h
#         segment_image = torch.Tensor(np.array(segment_image))
#         segment_image = torch.unsqueeze(segment_image, 0)
#         image = np.stack([t1_image, t1ce_image, t2_image, flair_image], axis=0)
#         image = einops.rearrange(image, 'b c w h -> (b c) w h')
#         # print(type(image), 'image')
#
#         # print( type(segment_image), 'segment_image')
#         return image/ 255, segment_image / 255

# class Get_tager_sample(Dataset):
#     def __init__(self, path):
#         self.path = path
#         self.name = os.listdir(os.path.join(path, 'flair'))
#
#     def __len__(self):
#         return len(self.name)
#
#     def __getitem__(self, index):
#         flair_name = self.name[index]
#         segment_path = os.path.join(self.path, 'seg', flair_name)
#         t1_path = os.path.join(self.path, 't1', flair_name)
#         t1ce_path = os.path.join(self.path, 't1ce', flair_name)
#         t2_path = os.path.join(self.path, 't2', flair_name)
#         flair_path = os.path.join(self.path, 'flair', flair_name)
#
#         # 加载图像
#         segment_image = keep_image_size_open(segment_path)  # 掩膜图像
#         t1_image = keep_image_size_open_rgb(t1_path)
#         t1_image = torch.Tensor(np.array(t1_image))
#         t1_image = einops.rearrange(t1_image, "w h c -> c w h")
#
#         t1ce_image = keep_image_size_open_rgb(t1ce_path)
#         t1ce_image = torch.Tensor(np.array(t1ce_image))
#         t1ce_image = einops.rearrange(t1ce_image, "w h c -> c w h ")
#
#         t2_image = keep_image_size_open_rgb(t2_path)
#         t2_image = torch.Tensor(np.array(t2_image))
#         t2_image = einops.rearrange(t2_image, "w h c -> c w h")
#
#         flair_image = keep_image_size_open_rgb(flair_path)
#         flair_image = torch.Tensor(np.array(flair_image))
#         flair_image = einops.rearrange(flair_image, "w h c -> c w h")
#
#         # 将所有模态堆叠为一个图像
#         image = np.stack([t1_image, t1ce_image, t2_image, flair_image], axis=0)
#         image = einops.rearrange(image, 'b c w h -> (b c) w h ')
#
#         # 处理分割掩膜（进行二分类处理）
#         segment_image = torch.Tensor(np.array(segment_image))
#         segment_image = torch.unsqueeze(segment_image, 0)
#
#         # 将掩膜中的非背景区域转换为1，背景为0
#         # 将所有非零的部分视为肿瘤区域
#         segment_image = torch.where(segment_image > 0, torch.tensor(1.0), torch.tensor(0.0))
#
#         # 返回数据（图像和二分类掩膜）
#         return image / 255, segment_image
def main(args):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training.")
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    print("设备", args.device)
    batch_size = args.batch_size
    # weights_path = args.weights
    args.lr *= args.world_size  # 学习率要根据并行GPU的数量进行倍增
    checkpoint_path = ""

    if args.local_rank == 0:  # 在第一个进程中打印信息，并实例化tensorboard
        print(args)
        print("initing over!")
        tb_write = SummaryWriter()
        print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')

    print(os.cpu_count())
    torch.cuda.empty_cache()
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    if args.local_rank == 0:
        print('Using {} dataloader workers every process'.format(nw))
    print(torch.cuda.memory_allocated() / 1024 / 1024)
    # datasets = Get_tager_sample(r"/home/ps/test_code/data/brats_/brats/train")
    # total_size=len(datasets)
    # train_size=int(0.9*total_size)
    # val_size=total_size-train_size
    # generator=torch.Generator().manual_seed(42)
    # train_1,test_1=random_split(datasets,[train_size,val_size],generator=generator)
    #test_1 = Get_tager_sample(r"/home/ps/test_code/data/tnscui2020/train")
    train_1 = Get_tager_sample(r"/home/ps/test_code/data/tnscui2020/train")
    test_1 = Get_tager_sample(r"/home/ps/test_code/data/tnscui2020/test")
    # train_1 = BRATSDataset("/home/yp/diskdata/data/MICCAI_BraTS2020_TrainingData", test_flag=False)

    # print('-----------------------------------------------test_1-----------------------------------------------')
    # print(type(test_1))
    # print(test_1.size())
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_1, shuffle=True)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_1)
    train_batch_sampler = torch.utils.data.BatchSampler(
        train_sampler, batch_size, drop_last=True)
    test_batch_sampler = torch.utils.data.BatchSampler(
        test_sampler, batch_size, drop_last=True)
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    if args.local_rank == 0:
        print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_1,
                                               batch_sampler=train_batch_sampler,
                                               pin_memory=True,
                                               num_workers=nw,
                                               )
    test_loader = torch.utils.data.DataLoader(test_1,
                                              batch_sampler=test_batch_sampler,
                                              pin_memory=True,
                                              num_workers=nw,
                                              )

    #########################网络
    # image, label = next(iter(train_loader))

    # 单个时刻图像
    net = UNet_conditional(c_in=1, c_out=1, device=device,con_c_in=3).to(device)
    # 多个时刻图像
    #net = UNet_conditional(c_in=1, c_out=1, device=device, con_c_in=12).to(device)
    checkpoint_path = r"initial_weights.pt"
    if args.local_rank == 0:
        torch.save(net.state_dict(), checkpoint_path)
    # checkpoint_path = r"/home/ps/test_code/shiyan/code/bratsseg_base/chest2_model/min_loss:2.66502346843481060.pth"
    # weight_dict = torch.load(checkpoint_path, map_location='cpu')
    # load_weight_dict = {k: v for k, v in weight_dict.items()
    #                     if net.state_dict()[k].numel() == v.numel()}

    dist.barrier()
    if args.local_rank == 0:
        myeam = EMA(net,beta=0.9999,update_every=1).to("cuda:0")
    net.load_state_dict(torch.load(checkpoint_path, map_location=device),strict=False)
    #net.load_state_dict(load_weight_dict, strict=False)
    dist.barrier()

    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.local_rank], find_unused_parameters=True)
    diffusion = Diffusion(img_size=256, device=device)
    lr = 1e-4
    net_optim = optim.AdamW(net.parameters(), lr=lr,weight_decay=1e-5)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=net_optim, T_max=400, eta_min=1e-6, last_epoch=-1)
    # lr = 1e-4
    # net_optim = optim.Adam(net.parameters(), lr=lr)
    # cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
    #         optimizer=net_optim, T_max=800, eta_min=1e-6, last_epoch=-1)
    #scalar = GradScaler()
    min_loss=999
    for epoch in range(800):
        torch.cuda.empty_cache()
        train_sampler.set_epoch(epoch)

        loss_mean = train(device=device, model=net, optimizer=net_optim, diffusion=diffusion, dataloader=train_loader,
                          epoch=epoch)
        val_loss_mean = val(device=device, model=net, diffusion=diffusion, dataloader=test_loader,epoch=epoch)
        cosineScheduler.step()
        if args.local_rank == 0:
            myeam.update()
            tb_write.add_scalar("mes", 1, epoch)
            if (epoch % 30 == 0):
                # net = torch.nn.parallel.DistributedDataParallel(net, device_ids= 0, find_unused_parameters=True)
                img, lable = next(iter(test_loader))
                sampled_images = diffusion.sample_condition_decoder(net, n=img.shape[0], image=img)
                # numpy.save("1.npy",sampled_images.cpu().numpy())
                sampled_images_show = (sampled_images.clamp(-1, 1) + 1) / 2
                sampled_images_show = (sampled_images_show * 255).type(torch.uint8)
                save_images_double(sampled_images_show, os.path.join("train_result", "train_brats", f"{epoch}.jpg"),
                                   lable)
                save_images(sampled_images, os.path.join("train_result", "train_brats_single", f"{epoch}.jpg"))
                save_images(lable, os.path.join("train_result", "train_brats_label", f"{epoch}.jpg"))
                save_images_single_channel(sampled_images,
                                           os.path.join("train_result", "train_brats_image_single_channel"), epoch)
                save_images_single_channel(lable, os.path.join("train_result", "train_brats_label_single_channel"),
                                           epoch)

            #torch.save(net.module.state_dict()  , "chest_model/" + f"{epoch}.pth")
            if val_loss_mean<min_loss:
                print(f"val_loss_mean:{val_loss_mean:.5f}",f"min_loss:{min_loss:.5f}")
                torch.save(myeam.ema_model.state_dict(), "chest2_model/" +f"min_loss:{val_loss_mean:.5f}," +f"{epoch}.pth")
                min_loss = val_loss_mean
    cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=50)
    ## base batch-size = 37 limanliuxing batch-size = 20
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
    parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)
    opt = parser.parse_args()

    main(opt)



# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --use_env ddpm2.py
