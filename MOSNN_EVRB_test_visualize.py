import argparse
import os
import time
import glob
import random

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import numpy as np
from PIL import Image
from torchvision import transforms
from models_CTSN.fusion_models import Fusion_TernarySpike

from models.fusion_models import Fusion_MOSNN
from functions import seed_all
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

# 显卡设置
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser(description='PyTorch EVRB Testing')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('-b', '--batch-size', default=1, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('-p', '--print-freq', default=10, type=int, metavar='N',
                    help='print frequency (default: 10)')
parser.add_argument('--seed', default=600, type=int, help='seed for initializing testing.')
parser.add_argument('--T', default=10, type=int, metavar='N', help='snn simulation time (default: 10)')

# 模型加载路径
parser.add_argument('--load', type=str, required=True, help='/home/zy/data/zy/zhaoyue/MotionSNN-EVRB/mosnn_best_31.1330_0.9375_0504_evrb_T10_model_distribute.pth')

# 保存结果
parser.add_argument('--save-results', type=bool, default=True, help='whether to save output images')
parser.add_argument('--results-dir', type=str, default='results_EVRB_TernarySpike', help='dir to save output images')

# EVRB数据集路径参数 (与训练脚本保持一致)
parser.add_argument('--evrb-root', type=str, default='/home/zy/data/zy/zhaoyue/Datasets/EVRB_SNN',
                    help='EVRB 根目录')
parser.add_argument('--evrb-test-dir', type=str, default='test', help='EVRB 测试集子目录名')
parser.add_argument('--evrb-eventframes-subdir', type=str, default='event_frames_T10', help='事件帧目录名')
parser.add_argument('--evrb-blur-subdir', type=str, default='blur_processed', help='模糊图目录名')
parser.add_argument('--evrb-gt-subdir', type=str, default='gt_processed', help='GT图目录名')
parser.add_argument('--evrb-max-height', type=int, default=0, help='读取 EVRB 时沿高度方向截取的最大高度')


# ------------------ EVRB Dataloader 开始 ------------------
def read_rgb_np(p):
    return np.array(Image.open(p).convert("RGB"))

class EVRBEventFramesFolderDataset(torch.utils.data.Dataset):
    def __init__(self, split_root, eventframes_subdir="event_frames_T10", blur_subdir="blur_processed",
                 gt_subdir="gt_processed", cropsize=0, datarand=False, max_height=0, is_train=False):
        self.split_root = split_root
        self.eventframes_subdir = eventframes_subdir
        self.blur_subdir = blur_subdir
        self.gt_subdir = gt_subdir
        self.cropsize = int(cropsize) if cropsize else 0
        self.datarand = datarand
        self.max_height = int(max_height) if max_height else 0
        self.is_train = is_train
        self.to_tensor = transforms.ToTensor()

        self.items = []

        if os.path.isdir(split_root):
            for seq_name in sorted(os.listdir(split_root)):
                seq_dir = os.path.join(split_root, seq_name)
                if not os.path.isdir(seq_dir):
                    continue

                frames_dir = os.path.join(seq_dir, self.eventframes_subdir)
                if not os.path.isdir(frames_dir):
                    continue

                pt_files = sorted(glob.glob(os.path.join(frames_dir, "*.pt")))
                for pt_path in pt_files:
                    frame_base = os.path.splitext(os.path.basename(pt_path))[0]
                    blur_path = os.path.join(seq_dir, self.blur_subdir, f"{frame_base}.png")
                    sharp_path = os.path.join(seq_dir, self.gt_subdir, f"{frame_base}.png")
                    if not os.path.exists(blur_path) or not os.path.exists(sharp_path):
                        continue
                    self.items.append((seq_dir, frame_base, pt_path))

        if not self.items:
            raise FileNotFoundError(f"No valid samples found in: {split_root}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        if self.datarand:
            seq_dir, frame_base, pt_path = self.items[random.randint(0, len(self.items) - 1)]
        else:
            seq_dir, frame_base, pt_path = self.items[index]

        sample_name = f"{os.path.basename(seq_dir)}/{frame_base}"

        spikes = torch.load(
            pt_path,
            map_location="cpu",
        )
        if isinstance(spikes, np.ndarray):
            spikes = torch.from_numpy(spikes)
        spikes = spikes.float()

        blur_path = os.path.join(seq_dir, self.blur_subdir, f"{frame_base}.png")
        sharp_path = os.path.join(seq_dir, self.gt_subdir, f"{frame_base}.png")

        blur_np = read_rgb_np(blur_path)
        sharp_np = read_rgb_np(sharp_path)

        if self.max_height:
            blur_np = blur_np[:self.max_height, :, :]
            sharp_np = sharp_np[:self.max_height, :, :]
            spikes = spikes[:, :, :self.max_height, :]

        blur_t = self.to_tensor(blur_np)
        sharp_t = self.to_tensor(sharp_np)

        # 补全了原版代码中的 cropsize 逻辑
        if self.cropsize:
            _, H, W = sharp_t.shape
            ps = self.cropsize
            if H >= ps and W >= ps:
                if self.is_train:
                    rr = random.randint(0, H - ps)
                    cc = random.randint(0, W - ps)
                else:
                    # 测试时使用中心裁剪
                    rr = (H - ps) // 2
                    cc = (W - ps) // 2
                blur_t = blur_t[:, rr:rr+ps, cc:cc+ps]
                sharp_t = sharp_t[:, rr:rr+ps, cc:cc+ps]
                spikes = spikes[:, :, rr:rr+ps, cc:cc+ps]

        return spikes, blur_t, sharp_t, sample_name
# ------------------ EVRB Dataloader 结束 ------------------


def main():
    args = parser.parse_args()
    args.nprocs = torch.cuda.device_count()
    if args.nprocs < 1:
        raise RuntimeError("No CUDA device found.")
    
    mp.spawn(main_worker, nprocs=args.nprocs, args=(args.nprocs, args))


def main_worker(local_rank, nprocs, args):
    args.local_rank = local_rank

    if args.seed is not None:
        seed_all(args.seed + local_rank)

    cudnn.benchmark = True

    dist.init_process_group(backend='nccl',
                            init_method='tcp://127.0.0.1:2456',
                            world_size=args.nprocs,
                            rank=local_rank)

    model = Fusion_TernarySpike(imgchannel=3, eventchannel=2, outchannel=3)
    model.T = args.T

    if args.local_rank == 0:
        total = sum([param.nelement() for param in model.parameters()])
        print("Model params is {:.4f} MB".format(total / 1e6))

    if args.load is not None:
        if args.local_rank == 0:
            print(f"=> Loading checkpoint: {args.load}")
            
        # 1. 先把权重文件读出来
        checkpoint = torch.load(args.load, map_location='cuda:{}'.format(args.local_rank))
        
        # 2. 兼容有的代码会把整个训练状态（epoch, optimizer等）打包成字典保存的情况
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        # 3. 遍历字典，把 'module.' 前缀清理干净
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k  # 去掉 `module.`
            new_state_dict[name] = v
            
        # 4. 强制严格加载 (strict=True)，如果有对不上的直接暴露问题！
        model.load_state_dict(new_state_dict, strict=True)
        
    else:
        raise ValueError("Please specify a model path using --load")

    torch.cuda.set_device(local_rank)
    model.cuda(local_rank)

    args.batch_size = max(1, int(args.batch_size / args.nprocs))
    model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[local_rank],
                                                      find_unused_parameters=True)

    # 准备测试数据集 (EVRB)
    val_root = os.path.join(args.evrb_root, args.evrb_test_dir)
    val_dataset = EVRBEventFramesFolderDataset(
        split_root=val_root,
        eventframes_subdir=args.evrb_eventframes_subdir,
        blur_subdir=args.evrb_blur_subdir,
        gt_subdir=args.evrb_gt_subdir,
        cropsize=0,
        datarand=False,
        max_height=args.evrb_max_height,
        is_train=False,
    )

    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             num_workers=args.workers,
                                             pin_memory=True,
                                             sampler=val_sampler)

    # 创建保存目录
    if args.save_results and args.local_rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        print(f"=> Output images will be saved to: {args.results_dir}")

    val_PSNR, val_SSIM = validate(val_loader, model, local_rank, args)

    if local_rank == 0:
        print('################################### test end #########################################')
        print('Test EVRB: PSNR: {:.4f}, SSIM: {:.4f}'.format(val_PSNR, val_SSIM))


def validate(val_loader, model, local_rank, args):
    PSNR_list = []
    SSIM_list = []

    model.eval()
    start_t = time.time()
    if local_rank == 0:
        print("local rank {:} begin to validate...".format(local_rank))

    with torch.no_grad():
        for i, (inputSpikes, input_Img, gt_Img, sample_names) in enumerate(val_loader):
            t1 = time.time()
            inputSpikes = inputSpikes.cuda(local_rank, non_blocking=True)
            input_Img = input_Img.cuda(local_rank, non_blocking=True)
            gt_Img = gt_Img.cuda(local_rank, non_blocking=True)

            ## 计算网络输出
            output = model(inputSpikes, input_Img)

            ## 计算PSNR/SSIM并保存结果
            for res, tar, s_name in zip(output, gt_Img, sample_names):
                res_np = res.clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
                tar_np = tar.clamp(0, 1).cpu().numpy().transpose(1, 2, 0)

                # PSNR / SSIM 
                psnr = compare_psnr(tar_np, res_np, data_range=1.0)
                try:
                    # 兼容新版本 skimage 0.19+
                    ssim = compare_ssim(tar_np, res_np, multichannel=True, data_range=1.0)
                except TypeError:
                    # 兼容旧版本 skimage 0.18及之前
                    ssim = compare_ssim(tar_np, res_np, multichannel=True, data_range=1.0)

                PSNR_list.append(psnr)
                SSIM_list.append(ssim)

                if args.save_results:
                    # 【修改点】：在文件名末尾加上 PSNR 和 SSIM 的值
                    # 使用 .2f 和 .4f 来控制小数位数，防止文件名过长
                    file_name_with_metrics = f"{s_name}_PSNR_{psnr:.2f}_SSIM_{ssim:.4f}.png"
                    
                    save_path = os.path.join(args.results_dir, file_name_with_metrics)
                    
                    # 提取出子文件夹路径并确保它存在
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    
                    res_img = (res_np * 255.0).round().astype(np.uint8)
                    Image.fromarray(res_img).save(save_path)

            batch_t = time.time() - start_t
            val_t = time.time() - t1
            if local_rank == 0 and i % args.print_freq == 0:
                print('Testing index: {:} Test PSNR: {:.4f}   SSIM: {:.4f}   batch time: {:.2f}   val time: {:.2f}'.format(
                    i, np.mean(PSNR_list), np.mean(SSIM_list), batch_t, val_t))
            start_t = time.time()

        # 此处使用局部平均值，若想获取更精确的全局指标可以添加dist.all_reduce
        PSNR = np.mean(PSNR_list)
        SSIM = np.mean(SSIM_list)

    return PSNR, SSIM


if __name__ == '__main__':
    main()