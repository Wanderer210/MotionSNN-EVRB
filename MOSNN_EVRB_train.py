import argparse
import os
import time
import random
import warnings
import sys
import glob
from datetime import datetime

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from models.fusion_models import Fusion_TernarySpike
from functions import TET_loss, seed_all
from functions import CharbonnierLoss, EdgeLoss
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
parser = argparse.ArgumentParser(description='PyTorch Temporal Efficient Training')
parser.add_argument('-j',
                    '--workers',
                    default=8,
                    type=int,
                    metavar='N',
                    help='number of data loading workers; -1 means auto safe setting')
parser.add_argument('--epochs',
                    default=800,
                    type=int,
                    metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch',
                    default=0,
                    type=int,
                    metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b',
                    '--batch-size',
                    default=8,
                    type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr',
                    '--learning-rate',
                    default=1e-4,
                    type=float,
                    metavar='LR',
                    help='initial learning rate',
                    dest='lr')
parser.add_argument('-p',
                    '--print-freq',
                    default=20,
                    type=int,
                    metavar='N',
                    help='print frequency (default: 10)')
parser.add_argument('-e',
                    '--evaluate',
                    default=False,
                    dest='evaluate',
                    action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--seed',
                    default=600,
                    type=int,
                    help='seed for initializing training. ')
parser.add_argument('--T',
                    default=10,
                    type=int,
                    metavar='N',
                    help='snn simulation time (default: 2)')
parser.add_argument('--means',
                    default=1.0,
                    type=float,
                    metavar='N',
                    help='make all the potential increment around the means (default: 1.0)')
parser.add_argument('--TET',
                    default=True,
                    type=bool,
                    metavar='N',
                    help='if use Temporal Efficient Training (default: True)')
parser.add_argument('--lamb',
                    default=0.0,
                    type=float,
                    metavar='N',
                    help='adjust the norm factor to avoid outlier (default: 0.0)')
parser.add_argument('--outpath',
                    type=str,
                    default='MOSNN_out_EVRB',
                    help='output dir path')

parser.add_argument('--evrb-root',
                    type=str,
                    default='/home/star/zhaoyue/DirectionSNN-EVRB/EVRB',
                    help='EVRB 根目录（需包含 train/ 与 test/，每个序列为 00000/00001/... 子目录）')
parser.add_argument('--evrb-train-dir',
                    type=str,
                    default='train',
                    help='EVRB 训练集子目录名（默认: train）')
parser.add_argument('--evrb-test-dir',
                    type=str,
                    default='test',
                    help='EVRB 测试/验证集子目录名（默认: test）')
parser.add_argument('--evrb-eventframes-subdir',
                    type=str,
                    default='event_frames_T10',
                    help='每个序列内事件帧目录名（默认: event_frames_T10，文件为 *.pt）')
parser.add_argument('--evrb-blur-subdir',
                    type=str,
                    default='blur_processed',
                    help='每个序列内模糊图目录名（默认: blur_processed，文件为 *.png）')
parser.add_argument('--evrb-gt-subdir',
                    type=str,
                    default='gt_processed',
                    help='每个序列内清晰 GT 图目录名（默认: gt_processed，文件为 *.png）')
parser.add_argument('--evrb-max-height',
                    type=int,
                    default=0,
                    help='读取 EVRB 时沿高度方向截取的最大高度；0 表示不截断（保持原始分辨率）')
parser.add_argument('--crop',
                    type=int,
                    default=0,
                    help='训练阶段随机裁剪大小；0 表示不裁剪')
parser.add_argument('--spike-monitor-freq',
                    default=0,
                    type=int,
                    metavar='N',
                    help='训练阶段监控脉冲比例的步频；0 表示关闭')
parser.add_argument('--spike-vis-freq',
                    default=0,
                    type=int,
                    metavar='N',
                    help='训练阶段保存脉冲可视化图的步频；0 表示关闭')
parser.add_argument('--spike-vis-layer',
                    default='s1',
                    type=str,
                    help='保存可视化时选择的脉冲层: s1/s2/s3/s4')
args = parser.parse_args()
#args.resume = '/home/zy/zhaoyue/DirectionSNN-Gopro/MOSNN_out_SAT_direction_LIF/MOSNN_GoPro_20260123_224436/checkpoints/epoch_0220_psnr_37868.4805_gopro_T10_model_distribute.pth'
args.resume = None


def get_safe_num_workers(requested_workers, nprocs):
    cpu_cnt = os.cpu_count() or 4
    if requested_workers is not None and requested_workers >= 0:
        return requested_workers
    auto_workers = min(4, max(1, cpu_cnt // (nprocs * 2)))
    return auto_workers


class SpikeVisualizer:
    @staticmethod
    def save_spike_distribution(spikes, save_path, title='spike', b=0, t=0, c=0):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np

        if hasattr(spikes, 'detach'):
            s = spikes
            if s.dim() == 5:
                s = s[b, t, c]
            elif s.dim() == 4:
                s = s[b, t]
            elif s.dim() == 3:
                s = s[t]
            elif s.dim() != 2:
                raise ValueError(f"Unsupported spike dim: {s.dim()}")
            s = s.detach().float().cpu().numpy()
        else:
            s = np.asarray(spikes)

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))

        im1 = axes[0, 0].imshow(s, cmap='RdBu_r', vmin=-1.5, vmax=1.5)
        axes[0, 0].set_title('spike value')
        plt.colorbar(im1, ax=axes[0, 0])

        im2 = axes[0, 1].imshow(np.sign(s), cmap='RdBu_r', vmin=-1, vmax=1)
        axes[0, 1].set_title('spike sign')
        plt.colorbar(im2, ax=axes[0, 1])

        dens = (np.abs(s) > 0.5).astype(np.float32)
        im3 = axes[1, 0].imshow(dens, cmap='binary', vmin=0, vmax=1)
        axes[1, 0].set_title('spike density')
        plt.colorbar(im3, ax=axes[1, 0])

        axes[1, 1].hist(s.flatten(), bins=50, alpha=0.7, color='steelblue')
        axes[1, 1].axvline(x=0.5, color='red', linestyle='--')
        axes[1, 1].axvline(x=-0.5, color='blue', linestyle='--')
        axes[1, 1].set_title('hist')
        axes[1, 1].grid(True, alpha=0.3)

        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    @staticmethod
    def save_input_gt_panel(input_img, gt_img, save_path, title='img', b=0):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        def _to_hwc_uint8(x):
            x = x.detach().float().cpu()
            if x.dim() == 4:
                x = x[b]
            if x.size(0) == 1:
                x = x.repeat(3, 1, 1)
            x = x.clamp(0.0, 1.0)
            x = (x * 255.0).round().byte().permute(1, 2, 0).numpy()
            return x

        inp = _to_hwc_uint8(input_img)
        gt = _to_hwc_uint8(gt_img)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        axes[0].imshow(inp)
        axes[0].set_title('input')
        axes[0].axis('off')

        axes[1].imshow(gt)
        axes[1].set_title('gt')
        axes[1].axis('off')

        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    @staticmethod
    def save_event_frame(event_frames, save_path, title='event', b=0, t=0):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np

        if not hasattr(event_frames, 'detach'):
            raise ValueError('event_frames must be a torch Tensor')

        e = event_frames.detach().float().cpu()
        if e.dim() == 5:
            e = e[b]
        if e.dim() != 4 or e.size(1) != 2:
            raise ValueError(f"Unsupported event_frames shape: {tuple(e.shape)}")

        T = int(e.size(0))
        t = int(max(0, min(t, T - 1)))

        pos = e[t, 0].numpy()
        neg = e[t, 1].numpy()
        signed = pos - neg
        dens = ((pos + neg) > 0).astype(np.float32)

        def _pmax(x, p=99.0):
            v = float(np.percentile(np.abs(x), p))
            return v if v > 0 else 1.0

        pos_vmax = _pmax(pos)
        neg_vmax = _pmax(neg)
        s_vmax = _pmax(signed)

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))

        im1 = axes[0, 0].imshow(pos, cmap='magma', vmin=0.0, vmax=pos_vmax)
        axes[0, 0].set_title('pos')
        axes[0, 0].axis('off')
        plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)

        im2 = axes[0, 1].imshow(neg, cmap='magma', vmin=0.0, vmax=neg_vmax)
        axes[0, 1].set_title('neg')
        axes[0, 1].axis('off')
        plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)

        im3 = axes[1, 0].imshow(signed, cmap='RdBu_r', vmin=-s_vmax, vmax=s_vmax)
        axes[1, 0].set_title('signed')
        axes[1, 0].axis('off')
        plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)

        im4 = axes[1, 1].imshow(dens, cmap='binary', vmin=0.0, vmax=1.0)
        axes[1, 1].set_title('density')
        axes[1, 1].axis('off')
        plt.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)

        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    @staticmethod
    def save_event_frames_per_t(event_frames, save_dir, prefix='event', b=0):
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        e = event_frames.detach()
        if e.dim() == 5:
            e = e[:1]
        if e.dim() == 4:
            T = int(e.size(0))
        elif e.dim() == 5:
            T = int(e.size(1))
        else:
            T = int(event_frames.size(1))

        for t in range(T):
            save_path = os.path.join(save_dir, f"{prefix}_t{t:02d}.png")
            SpikeVisualizer.save_event_frame(event_frames, save_path, title=f"{prefix} t{t}", b=b, t=t)


# 添加日志记录类
class TeeLogger:
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'a', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # 确保实时写入

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()

#分布式训练中计算损失函数或指标的全局平均值
def reduce_mean(tensor, nprocs):
    if nprocs <= 1 or (not dist.is_available()) or (not dist.is_initialized()):
        return tensor
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt

def read_rgb_np(p):
    return np.array(Image.open(p).convert("RGB"))

class EVRBEventFramesFolderDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        split_root,
        eventframes_subdir="event_frames_T10",
        blur_subdir="blur_processed",
        gt_subdir="gt_processed",
        cropsize=0,
        datarand=False,
        max_height=0,
        is_train=True,
    ):
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

        if self.cropsize:
            _, H, W = sharp_t.shape
            ps = self.cropsize
            if H >= ps and W >= ps:
                if self.is_train:
                    rr = random.randint(0, H - ps)
                    cc = random.randint(0, W - ps)
                else:
                    rr = (H - ps) // 2
                    cc = (W - ps) // 2
                blur_t = blur_t[:, rr:rr+ps, cc:cc+ps]
                sharp_t = sharp_t[:, rr:rr+ps, cc:cc+ps]
                spikes = spikes[:, :, rr:rr+ps, cc:cc+ps]

        return spikes, blur_t, sharp_t, sample_name


def compute_psnr_ssim_batch(output_t: torch.Tensor, gt_t: torch.Tensor):
    """
    output_t, gt_t: [N,C,H,W] torch tensors (GPU).
    """
    out = output_t.detach().cpu().numpy()
    gt = gt_t.detach().cpu().numpy()

    psnrs, ssims = [], []
    for res, tar in zip(out, gt):
        # NCHW -> HWC
        res = np.transpose(res, (1, 2, 0))
        tar = np.transpose(tar, (1, 2, 0))

        psnr = compare_psnr(tar, res)
        ssim = compare_ssim(tar, res, multichannel=True)
        psnrs.append(psnr)
        ssims.append(ssim)

    return (float(np.mean(psnrs)) if psnrs else 0.0,
            float(np.mean(ssims)) if ssims else 0.0)


def compute_lpips_batch(output_t: torch.Tensor, gt_t: torch.Tensor, lpips_fn):
    out = output_t.detach().clamp(0.0, 1.0)
    gt = gt_t.detach().clamp(0.0, 1.0)

    if out.size(1) == 1:
        out = out.repeat(1, 3, 1, 1)
    if gt.size(1) == 1:
        gt = gt.repeat(1, 3, 1, 1)

    out = out.float() * 2.0 - 1.0
    gt = gt.float() * 2.0 - 1.0

    return float(lpips_fn(out, gt).mean().item())


def main():
    # args = parser.parse_args()
    args.nprocs = torch.cuda.device_count()
    if args.nprocs < 1:
        raise RuntimeError("No CUDA device found.")

    args.workers = get_safe_num_workers(args.workers, args.nprocs)

    print(f"[Main] detected GPUs: {args.nprocs}")
    print(f"[Main] safe num_workers: {args.workers}")

    if args.nprocs == 1:
        main_worker(0, 1, args)
    else:
        mp.spawn(main_worker, nprocs=args.nprocs, args=(args.nprocs, args), join=True)

def main_worker(local_rank, nprocs, args):
    args.local_rank = local_rank

    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')

    checkpoint_name = None
    if not checkpoint_name:
        checkpoint_name = "MOSNN_EVRB_" + time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_path = os.path.join(args.outpath, checkpoint_name, 'logs')
    checkpoint_path = os.path.join(args.outpath, checkpoint_name, 'checkpoints')

    # 设置训练日志文件路径
    training_log_path = os.path.join(args.outpath, checkpoint_name, 'training_logs')

    writer = None
    tee_logger = None
    spike_vis_dir = os.path.join(args.outpath, checkpoint_name, 'spike_vis')

    if args.local_rank == 0:
        os.makedirs(args.outpath, exist_ok=True)
        os.makedirs(log_path, exist_ok=True)
        os.makedirs(checkpoint_path, exist_ok=True)
        os.makedirs(training_log_path, exist_ok=True)
        if args.spike_vis_freq:
            os.makedirs(spike_vis_dir, exist_ok=True)

        # 创建训练日志文件
        log_filename = os.path.join(training_log_path, f'training_log_{checkpoint_name}.txt')

        # 重定向stdout到日志文件和终端
        tee_logger = TeeLogger(log_filename)
        sys.stdout = tee_logger

        writer = SummaryWriter(log_path, purge_step=args.start_epoch)
        writer.add_text('checkpoint', checkpoint_name)

        # 记录训练开始信息
        print(f"summary path: {log_path}")
        print(f"=== 训练开始 ===")
        print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"检查点名称: {checkpoint_name}")
        print(f"日志文件: {log_filename}")
        print(f"总epochs: {args.epochs}")
        print(f"批次大小: {args.batch_size}")
        print(f"学习率: {args.lr}")
        print(f"GPU数量: {nprocs}")
        print(f"="*50)

    if args.seed is not None:
        seed_all(args.seed + local_rank)

    cudnn.deterministic = False
    cudnn.benchmark = True

    if local_rank == 0:
        warnings.warn(
            'Using cudnn.benchmark=True and deterministic=False for safer multi-GPU training.'
        )

    if nprocs > 1:
        dist.init_process_group(backend='nccl',
                                init_method='tcp://127.0.0.1:2456',
                                world_size=nprocs,
                                rank=local_rank)

    load_names = None   #当前不加载预训练模型（从头开始训练）
    save_names = f'evrb_T{args.T}_model_distribute.pth'

    #load_names = os.path.join(args.outpath, 'MOSNN_GoPro_20250816_180919', 'checkpoints', 'epoch_0300_psnr_31.6874_gopro_T10_model_distribute.pth')
    # 当需要从检查点恢复训练时，将 load_names 设置为具体的模型文件路径

    model = Fusion_MOSNN(imgchannel=3, eventchannel=2, outchannel=3)
    model.T = args.T


    #模型参数统计
    if args.local_rank == 0:
        if hasattr(model, 'count_params_split'):
            split = model.count_params_split(trainable_only=False)
            print(
                "Model params (M): total={:.4f}, snn={:.4f}, cnn={:.4f}".format(
                    split['total'] / 1e6,
                    split['snn'] / 1e6,
                    split['cnn'] / 1e6,
                )
            )
        else:
            total = sum([param.nelement() for param in model.parameters()])
            print("Model params is {:.4f} MB".format(total / 1e6))  #模型的总参数量

    model = model.to(device)

    #预训练模型加载
    if load_names != None:
        if args.local_rank == 0 and writer is not None:
            writer.add_text('load_data', load_names)
        model.load_state_dict(torch.load(load_names, map_location=device), strict=False)

    args.batch_size = max(1, int(args.batch_size / nprocs))

    if nprocs > 1:
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[local_rank],
                                                          output_device=local_rank,
                                                          find_unused_parameters=False,
                                                          broadcast_buffers=False)

    # 定义损失函数和优化器
    criterion_char = CharbonnierLoss()
    criterion_edge = EdgeLoss().to(device)
    if hasattr(criterion_edge, 'kernel') and criterion_edge.kernel is not None:
        criterion_edge.kernel = criterion_edge.kernel.to(device)

    lpips_fn = None

    #优化器和学习率调度器配置
    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)

    for pg in optimizer.param_groups:
        if 'initial_lr' not in pg:
            pg['initial_lr'] = pg['lr']

    eta_min = 1e-6
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs - args.start_epoch),
        eta_min=eta_min,
        last_epoch=-1,
    )

    # Data loading code (EVRB)
    if not args.evrb_root:
        raise ValueError("EVRB root is not set. Please provide --evrb-root")

    train_root = os.path.join(args.evrb_root, args.evrb_train_dir)
    val_root = os.path.join(args.evrb_root, args.evrb_test_dir)

    train_dataset = EVRBEventFramesFolderDataset(
        split_root=train_root,
        eventframes_subdir=args.evrb_eventframes_subdir,
        blur_subdir=args.evrb_blur_subdir,
        gt_subdir=args.evrb_gt_subdir,
        cropsize=args.crop,
        datarand=False,
        max_height=args.evrb_max_height,
        is_train=True,
    )
    val_dataset = EVRBEventFramesFolderDataset(
        split_root=val_root,
        eventframes_subdir=args.evrb_eventframes_subdir,
        blur_subdir=args.evrb_blur_subdir,
        gt_subdir=args.evrb_gt_subdir,
        cropsize=args.crop,
        datarand=False,
        max_height=args.evrb_max_height,
        is_train=False,
    )

    if nprocs > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=nprocs, rank=local_rank, shuffle=True, drop_last=False)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=nprocs, rank=local_rank, shuffle=False, drop_last=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=(train_sampler is None),
                                               num_workers=args.workers,
                                               pin_memory=True,
                                               sampler=train_sampler,
                                               persistent_workers=(args.workers > 0),
                                               drop_last=False)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True,
                                             sampler=val_sampler,
                                             persistent_workers=(args.workers > 0),
                                             drop_last=False)

    best_psnr = 0.
    best_ssim = 0.
    val_PSNR = 0.
    val_SSIM = 0.

    best_save_state = None
    best_save_epoch = -1
    best_save_psnr = -1.0
    best_save_ssim = 0.0

    last_improve_epoch = int(args.start_epoch)

    if args.resume:
        if os.path.isfile(args.resume):
            if local_rank == 0:
                print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=device)
            target_model = model.module if hasattr(model, "module") else model
            if 'epoch' in checkpoint:
                args.start_epoch = checkpoint['epoch']
                best_psnr = checkpoint['best_psnr']
                best_ssim = checkpoint['best_ssim']

                # --- 添加以下临时修复代码 ---
                if best_psnr > 100: # PSNR不可能超过100，如果是几千说明是旧Bug导致的
                    if local_rank == 0:
                        print(f"检测到异常的 best_psnr ({best_psnr})，正在重置为 0 以修复保存逻辑...")
                    best_psnr = 0.
                    best_ssim = 0.
                target_model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])

                if 'scheduler' in checkpoint:
                    try:
                        scheduler.load_state_dict(checkpoint['scheduler'])
                    except Exception:
                        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                            optimizer,
                            eta_min=1e-6,
                            T_max=max(1, args.epochs - args.start_epoch),
                            last_epoch=-1,
                        )

                if local_rank == 0:
                    print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
                    print("=> resume lr from checkpoint: {}".format(optimizer.state_dict()['param_groups'][0]['lr']))
            else:
                target_model.load_state_dict(checkpoint)
                if local_rank == 0:
                    print("=> loaded only model state_dict from '{}'. Optimizer and scheduler not loaded.".format(args.resume))
        else:
            if local_rank == 0:
                print("=> no checkpoint found at '{}'".format(args.resume))

    if args.evaluate:
        validate(val_loader, model, criterion_edge, criterion_char, local_rank, args)
        if nprocs > 1 and dist.is_initialized():
            dist.destroy_process_group()
        return

    # 添加训练时长估计相关变量
    epoch_times = []
    training_start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        epoch_start = time.time()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if val_sampler is not None:
            val_sampler.set_epoch(epoch)

        if local_rank == 0:
            print("epoch {:} start. lr={:} ".format(epoch, optimizer.state_dict()['param_groups'][0]['lr']))
            if writer is not None:
                writer.add_scalar('train/lr', optimizer.state_dict()['param_groups'][0]['lr'], epoch)

        # train for one epoch
        train_start = time.time()
        train_PSNR, train_SSIM, train_Loss = train(
            train_loader,
            model,
            criterion_edge,
            criterion_char,
            optimizer,
            local_rank,
            args,
            epoch,
            writer=writer,
            spike_vis_dir=spike_vis_dir,
        )
        train_t = time.time() - train_start

        if nprocs > 1:
            dist.barrier()

        if args.local_rank == 0 and writer is not None:
            writer.add_scalar('train/loss', train_Loss, epoch)
            writer.add_scalar('train/psnr', train_PSNR, epoch)
            writer.add_scalar('train/ssim', train_SSIM, epoch)
            writer.add_scalar('train/time', train_t, epoch)

        do_validate = ((epoch + 1) % 5 == 0) or (epoch == args.epochs - 1)
        if do_validate:
            val_start = time.time()
            val_PSNR, val_SSIM, val_Loss = validate(
                val_loader, model,
                criterion_edge, criterion_char,
                local_rank, args
            )
            val_t = time.time() - val_start

            if args.local_rank == 0 and writer is not None:
                writer.add_scalar('val/loss', val_Loss, epoch)
                writer.add_scalar('val/psnr', val_PSNR, epoch)
                writer.add_scalar('val/ssim', val_SSIM, epoch)
                writer.add_scalar('val/time', val_t, epoch)

        scheduler.step()

        # 计算当前epoch时间并更新时间估计
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        # 训练时长估计逻辑
        if local_rank == 0:
            # 计算平均epoch时间
            avg_epoch_time = np.mean(epoch_times)

            # 计算已用时间
            elapsed_time = time.time() - training_start_time

            # 计算剩余epoch数
            remaining_epochs = args.epochs - (epoch + 1)

            # 估计剩余时间
            estimated_remaining_time = remaining_epochs * avg_epoch_time

            # 估计总训练时间
            estimated_total_time = elapsed_time + estimated_remaining_time

            # 格式化时间显示
            def format_time(seconds):
                hours = int(seconds // 3600)
                minutes = int((seconds % 3600) // 60)
                secs = int(seconds % 60)
                return f"{hours:02d}:{minutes:02d}:{secs:02d}"

            # 打印时间估计信息
            print(f"=== 训练时长估计 ===")
            print(f"当前epoch时间: {format_time(epoch_time)}")
            print(f"平均epoch时间: {format_time(avg_epoch_time)}")
            print(f"已用时间: {format_time(elapsed_time)}")
            print(f"预计剩余时间: {format_time(estimated_remaining_time)}")
            print(f"预计总训练时间: {format_time(estimated_total_time)}")
            print(f"训练进度: {((epoch + 1) / args.epochs * 100):.1f}% ({epoch + 1}/{args.epochs})")

            # 记录到tensorboard
            if writer is not None:
                writer.add_scalar('time_estimation/avg_epoch_time', avg_epoch_time, epoch)
                writer.add_scalar('time_estimation/elapsed_time', elapsed_time, epoch)
                writer.add_scalar('time_estimation/estimated_remaining_time', estimated_remaining_time, epoch)
                writer.add_scalar('time_estimation/training_progress', (epoch + 1) / args.epochs * 100, epoch)

        if do_validate:
            is_best_psnr = val_PSNR > best_psnr
            is_best_ssim = val_SSIM > best_ssim
            if is_best_psnr:
                best_psnr = val_PSNR
            if is_best_ssim:
                best_ssim = val_SSIM

            improved_any = is_best_psnr or is_best_ssim
            if improved_any:
                last_improve_epoch = int(epoch)

            if local_rank == 0:
                print('Epoch {:} Best PSNR: {:.4f}, Best SSIM: {:.4f}, Val PSNR: {:.4f}, Val SSIM: {:.4f}'.format(epoch, best_psnr, best_ssim, val_PSNR, val_SSIM))
                print('################################### epoch end #########################################')

        if args.local_rank == 0:
            target_model = model.module if hasattr(model, "module") else model
            state_to_save = {
                'epoch': epoch + 1,
                'state_dict': target_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_psnr': best_psnr,
                'best_ssim': best_ssim,
            }

            if do_validate and is_best_psnr and save_names is not None:
                torch.save(state_to_save, os.path.join(checkpoint_path, 'best_{:.4f}_{:.4f}_{:04}_'.format(best_psnr, best_ssim, epoch) + save_names))

            if epoch % 10 == 0:
                torch.save(state_to_save, os.path.join(checkpoint_path, 'epoch_{:04}_psnr_{:.4f}_'.format(epoch, val_PSNR) + save_names))

        if do_validate:
            should_stop = False
            # 移除旧的缓存保存逻辑
            stop_t = torch.tensor(1 if should_stop else 0, device=device, dtype=torch.int32)
            if nprocs > 1:
                dist.broadcast(stop_t, src=0)
            if int(stop_t.item()) == 1:
                break

        if args.local_rank == 0 and writer is not None:
            writer.add_scalar('epoch_time', time.time() - epoch_start, epoch)



    # 训练结束时的处理
    if args.local_rank == 0:
        def format_time(seconds):
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"

        print(f"=== 训练完成 ===")
        print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"总训练时间: {format_time(time.time() - training_start_time)}")
        print(f"最佳PSNR: {best_psnr:.4f}")
        print(f"最佳SSIM: {best_ssim:.4f}")
        print(f"="*50)

        if tee_logger is not None:
            sys.stdout = tee_logger.terminal
            tee_logger.close()

        if writer is not None:
            writer.close()
        print("summary closed.")

    if nprocs > 1 and dist.is_initialized():
        dist.destroy_process_group()



def train(train_loader, model, criterion_edge, criterion_char, optimizer, local_rank, args, epoch, writer=None, spike_vis_dir=None):
    model.train()
    device = torch.device(f'cuda:{local_rank}')

    psnr_sum = 0.0
    ssim_sum = 0.0
    loss_sum = 0.0
    count = 0

    if local_rank == 0:
        print(f"local rank {local_rank} begin to training...")

    start_t = time.time()

    # 在训练循环开始前，清零梯度
    optimizer.zero_grad()

    for i, (inputSpikes, input_Img, gt_Img, inputIndex) in enumerate(train_loader):
        t1 = time.time()

        gt_Img = gt_Img.to(device, non_blocking=True)
        inputSpikes = inputSpikes.to(device, non_blocking=True)
        input_Img = input_Img.to(device, non_blocking=True)

        need_spike = (local_rank == 0) and bool(getattr(args, 'spike_monitor_freq', 0)) and (i % args.spike_monitor_freq == 0)
        need_vis = (
            (local_rank == 0)
            and bool(getattr(args, 'spike_vis_freq', 0))
            and (i % args.spike_vis_freq == 0)
            and spike_vis_dir is not None
        )

        spike_info = None
        if need_spike or need_vis:
            output, spike_info = model(
                inputSpikes,
                input_Img,
                return_spikes=True,
                spike_mode='sample' if need_vis else 'stats',
            )
        else:
            output = model(inputSpikes, input_Img)

        if spike_info is not None and local_rank == 0:
            stats = spike_info.get('stats', {})
            if stats:
                msg = [f"SpikeStats ep={epoch} it={i}"]
                for k in ['s1', 's2', 's3', 's4']:
                    if k in stats and stats[k]:
                        d = stats[k]
                        msg.append(
                            f"{k}: den={d.get('spike_density', 0.0):.4f} +={d.get('pos_ratio', 0.0):.4f} 0={d.get('zero_ratio', 0.0):.4f} -={d.get('neg_ratio', 0.0):.4f}"
                        )
                print(' | '.join(msg))

                if writer is not None:
                    step = epoch * max(1, len(train_loader)) + i
                    for k, d in stats.items():
                        if not d:
                            continue
                        writer.add_scalar(f"spike/{k}_density", d.get('spike_density', 0.0), step)
                        writer.add_scalar(f"spike/{k}_pos", d.get('pos_ratio', 0.0), step)
                        writer.add_scalar(f"spike/{k}_neg", d.get('neg_ratio', 0.0), step)
                        writer.add_scalar(f"spike/{k}_zero", d.get('zero_ratio', 0.0), step)

            if need_vis and 'sample' in spike_info:
                sample_dict = spike_info['sample']
                target_layer = getattr(args, 'spike_vis_layer', 's1')
                if target_layer in sample_dict:
                    spk_tensor = sample_dict[target_layer]
                    save_name = f"ep{epoch:03d}_it{i:05d}_{target_layer}.png"
                    SpikeVisualizer.save_spike_distribution(
                        spk_tensor,
                        os.path.join(spike_vis_dir, save_name),
                        title=f"Layer {target_layer} | Ep {epoch} It {i}"
                    )

                    img_save_name = f"ep{epoch:03d}_it{i:05d}_img.png"
                    SpikeVisualizer.save_input_gt_panel(
                        input_Img, gt_Img,
                        os.path.join(spike_vis_dir, img_save_name),
                        title=f"Input/GT | Ep {epoch} It {i}"
                    )

                    evt_save_dir = os.path.join(spike_vis_dir, f"ep{epoch:03d}_it{i:05d}_events")
                    SpikeVisualizer.save_event_frames_per_t(
                        inputSpikes,
                        evt_save_dir,
                        prefix='event'
                    )

        loss_char = criterion_char(output, gt_Img)
        loss_edge = criterion_edge(output, gt_Img)
        total_loss = loss_char + (0.05 * loss_edge)

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 20.0)
        optimizer.step()

        mean_psnr, mean_ssim = compute_psnr_ssim_batch(output, gt_Img)
        bs = int(output.shape[0])
        psnr_sum += mean_psnr * bs
        ssim_sum += mean_ssim * bs
        loss_sum += float(total_loss.detach().cpu().item()) * bs
        count += bs

        if local_rank == 0 and (((i + 1) % args.print_freq == 0) or ((i + 1) == len(train_loader))):
            batch_t = time.time() - start_t
            train_t = time.time() - t1
            cur_loss = loss_sum / max(count, 1)
            cur_psnr = psnr_sum / max(count, 1)
            cur_ssim = ssim_sum / max(count, 1)
            print('Training index: {:}  PSNR: {:.4f}   SSIM: {:.4f}   LOSS: {:.4f}   batch time: {:.2f}   train time: {:.2f}'.format(
                i, cur_psnr, cur_ssim, cur_loss, batch_t, train_t))

            start_t = time.time()

    # ---- all_reduce SUM across ranks ----
    psnr_sum_t = torch.tensor(psnr_sum, device=device)
    ssim_sum_t = torch.tensor(ssim_sum, device=device)
    loss_sum_t = torch.tensor(loss_sum, device=device)
    count_t = torch.tensor(count, device=device, dtype=torch.long)

    if args.nprocs > 1:
        dist.all_reduce(psnr_sum_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(ssim_sum_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(loss_sum_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(count_t, op=dist.ReduceOp.SUM)

    denom = max(int(count_t.item()), 1)
    global_psnr = (psnr_sum_t / denom).item()
    global_ssim = (ssim_sum_t / denom).item()
    global_loss = (loss_sum_t / denom).item()
    return global_psnr, global_ssim, global_loss


def validate(val_loader, model, criterion_edge, criterion_char, local_rank, args):
    """
    验证阶段：
    - batch_size=1（每GPU）
    - 指标：clamp[0,1] + PSNR/SSIM/LPIPS
    - ✅ 全局严格平均：sum/count all_reduce（避免 mean-of-mean）
    """
    model.eval()
    device = torch.device(f'cuda:{local_rank}')

    psnr_sum = 0.0
    ssim_sum = 0.0
    loss_sum = 0.0
    count = 0

    start_t = time.time()
    if local_rank == 0:
        print(f"local rank {local_rank} begin to validate...")

    with torch.no_grad():
        for i, (inputSpikes, input_Img, gt_Img, inputIndex) in enumerate(val_loader):
            t1 = time.time()

            inputSpikes = inputSpikes.to(device, non_blocking=True)
            input_Img = input_Img.to(device, non_blocking=True)
            gt_Img = gt_Img.to(device, non_blocking=True)

            output = model(inputSpikes, input_Img)

            mean_psnr, mean_ssim = compute_psnr_ssim_batch(output, gt_Img)

            loss_char = criterion_char(output, gt_Img)
            loss_edge = criterion_edge(output, gt_Img)
            loss = loss_char + (0.05 * loss_edge)

            bs = output.shape[0]
            psnr_sum += mean_psnr * bs
            ssim_sum += mean_ssim * bs
            loss_sum += float(loss.detach().cpu().item()) * bs
            count += bs



            batch_t = time.time() - start_t
            val_t = time.time() - t1
            if local_rank == 0 and i % args.print_freq == 0:
                cur_psnr = psnr_sum / max(count, 1)
                cur_ssim = ssim_sum / max(count, 1)
                cur_loss = loss_sum / max(count, 1)
                print('Testing index: {:}  PSNR: {:.4f}   SSIM: {:.4f}   LOSS: {:.4f}   batch time: {:.2f}   val time: {:.2f}'.format(
                    i, cur_psnr, cur_ssim, cur_loss, batch_t, val_t))

            start_t = time.time()

    # ---- all_reduce SUM across ranks ----
    psnr_sum_t = torch.tensor(psnr_sum, device=device)
    ssim_sum_t = torch.tensor(ssim_sum, device=device)
    loss_sum_t = torch.tensor(loss_sum, device=device)
    count_t = torch.tensor(count, device=device, dtype=torch.long)

    if args.nprocs > 1:
        dist.all_reduce(psnr_sum_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(ssim_sum_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(loss_sum_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(count_t, op=dist.ReduceOp.SUM)

    denom = max(int(count_t.item()), 1)
    global_psnr = (psnr_sum_t / denom).item()
    global_ssim = (ssim_sum_t / denom).item()
    global_loss = (loss_sum_t / denom).item()

    return global_psnr, global_ssim, global_loss



if __name__ == '__main__':
    main()
