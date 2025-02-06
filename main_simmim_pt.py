# --------------------------------------------------------
# SimMIM
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# Modified by Zhenda Xie
# --------------------------------------------------------

import os
import time
import argparse
import datetime
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.amp as amp
from timm.utils import AverageMeter

from config import get_config
from models import build_model
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils_simmim import load_checkpoint, save_checkpoint, get_grad_norm, auto_resume_helper, reduce_tensor

import wandb
import yaml
from pathlib import Path

# pytorch major version (1.x or 2.x)
PYTORCH_MAJOR_VERSION = int(torch.__version__.split('.')[0])


def parse_option():
    parser = argparse.ArgumentParser('SimMIM pre-training script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--split-path', type=str, help='path to dataset split')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--enable-amp', action='store_true')
    parser.add_argument('--disable-amp', action='store_false', dest='enable_amp')
    parser.set_defaults(enable_amp=True)
    parser.add_argument('--fused_window_process', action='store_true', help='activate fused window process')
    parser.add_argument('--fused_layernorm', action='store_true', help='activate fused layernorm')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--tag', help='tag of experiment')

    # distributed training
    # for pytorch >= 2.0, use `os.environ['LOCAL_RANK']` instead
    # (see https://pytorch.org/docs/stable/distributed.html#launch-utility)
    if PYTORCH_MAJOR_VERSION == 1:
        parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')

    args = parser.parse_args()

    config = get_config(args)

    return args, config


def train(config, logger, wandb_logger):
    data_loader_train, data_loader_val, _ = build_loader(config, simmim=True, is_pretrain=True)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config, is_pretrain=True)
    model.cuda()
    logger.info(str(model))

    optimizer = build_optimizer(config, model, simmim=True, is_pretrain=True)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)
    model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model_without_ddp, 'flops'):
        flops = model_without_ddp.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))
    scaler = amp.GradScaler('cuda')

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT, logger)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, scaler, logger)
        if config.EVAL_MODE:
            visualize(config, data_loader_val, model)
            return

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH + 1, config.TRAIN.EPOCHS + 1):
        data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(config, model, data_loader_train, optimizer, epoch, lr_scheduler, scaler, logger, wandb_logger)
        if dist.get_rank() == 0 and config.SAVE_FREQ > 0 and (epoch % config.SAVE_FREQ == 0 or epoch == config.TRAIN.EPOCHS):
            save_checkpoint(config, epoch, model_without_ddp, 0., optimizer, lr_scheduler, scaler, logger)

        loss = validate(config, data_loader_val, model, epoch, logger, wandb_logger)
        logger.info(f"Loss of the network on the {len(data_loader_val.dataset)} val images: {loss:.4f}")

        if config.TRAIN.LR_SCHEDULER.NAME == 'plateau':
            lr_scheduler.step(epoch, loss)

            if config.TRAIN.LR_SCHEDULER.EARLY_STOP and lr_scheduler.has_hit_min(epoch, loss):
                if dist.get_rank() == 0 and config.SAVE_FREQ > 0:
                    save_checkpoint(config, epoch, model_without_ddp, 0., optimizer, lr_scheduler, scaler, logger)
                break

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def train_one_epoch(config, model, data_loader, optimizer, epoch, lr_scheduler, scaler, logger, wandb_logger):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    loss_scale_meter = AverageMeter()

    accum_loss = 0.0

    start = time.time()
    end = time.time()
    for idx, (img_lr, img_vhr, mask_lr) in enumerate(data_loader, start=1):
        img_lr = img_lr.cuda(non_blocking=True)
        img_vhr = img_vhr.cuda(non_blocking=True)
        mask_lr = mask_lr.cuda(non_blocking=True)

        with amp.autocast('cuda', enabled=config.ENABLE_AMP):
            _, _, loss = model(img_lr, img_vhr, mask_lr)

        if torch.isnan(loss).any():
            raise Exception('Loss is NaN')

        loss = loss / config.TRAIN.ACCUMULATION_STEPS
        accum_loss += loss.item()
        scaler.scale(loss).backward()

        if idx % config.TRAIN.ACCUMULATION_STEPS == 0:
            if config.TRAIN.CLIP_GRAD:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
            else:
                grad_norm = get_grad_norm(model.parameters())
            scaler.step(optimizer)
            optimizer.zero_grad()
            scaler.update()
            lr_scheduler.step_update(epoch * num_steps + idx)

            torch.cuda.synchronize()

            loss_meter.update(accum_loss, img_lr.size(0))
            norm_meter.update(grad_norm)
            loss_scale_meter.update(scaler.get_scale())
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % config.PRINT_FREQ == 0 or idx == len(data_loader):
                lr = optimizer.param_groups[0]['lr']
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                etas = batch_time.avg * (num_steps - idx)
                logger.info(
                    f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                    f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                    f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                    f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                    f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                    f'loss_scale {loss_scale_meter.val:.4f} ({loss_scale_meter.avg:.4f})\t'
                    f'mem {memory_used:.0f}MB')
            
            accum_loss = 0.0
            
    if dist.get_rank() == 0:
        wandb_logger.log(
            {
                'lr': lr,
                'train_loss': loss_meter.avg,
                'grad_norm': norm_meter.avg,
                'loss_scale': loss_scale_meter.avg
            },
            step=epoch
        )

    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


@torch.no_grad()
def validate(config, data_loader, model, epoch, logger, wandb_logger):
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()

    end = time.time()
    for idx, (img_lr, img_vhr, mask_lr) in enumerate(data_loader, start=1):
        img_lr = img_lr.cuda(non_blocking=True)
        img_vhr = img_vhr.cuda(non_blocking=True)
        mask_lr = mask_lr.cuda(non_blocking=True)

        with amp.autocast('cuda', enabled=config.ENABLE_AMP):
            _, _, loss = model(img_lr, img_vhr, mask_lr)

        loss = reduce_tensor(loss)

        loss_meter.update(loss.item(), img_lr.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0 or idx == len(data_loader):
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Val: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Mem {memory_used:.0f}MB')
            
    if dist.get_rank() == 0:
        wandb_logger.log(
            {
                'val_loss': loss_meter.avg
            },
            step=epoch
        )

    return loss_meter.avg


@torch.no_grad()
def visualize(config, data_loader, model):
    from torchvision.utils import save_image
    from tqdm import tqdm
    
    model.eval()

    output_folder = Path(config.MODEL.RESUME).with_suffix('')

    for i, (img_lr, img_vhr, mask_lr) in tqdm(enumerate(data_loader), 'Batches', total=len(data_loader)):
        img_lr = img_lr.cuda(non_blocking=True)
        img_vhr = img_vhr.cuda(non_blocking=True)
        mask_lr = mask_lr.cuda(non_blocking=True)

        x_vhr, x_vhr_rec, _ = model(img_lr, img_vhr, mask_lr)

        mean = torch.tensor(data_loader.dataset.transform.transform_compose.transforms[1].mean).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).cuda()
        std = torch.tensor(data_loader.dataset.transform.transform_compose.transforms[1].std).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).cuda()

        for j, (_x_vhr, _x_vhr_rec) in enumerate(zip(x_vhr, x_vhr_rec)):
            file_name_prefix = f'{i * len(x_vhr) + j}'

            file_name_input = output_folder / f'{file_name_prefix}_input.jpg'
            file_name_output = output_folder / f'{file_name_prefix}_output.jpg'
            file_name_both = output_folder / f'{file_name_prefix}_both.jpg'

            _x_vhr_both = (_x_vhr + _x_vhr_rec) * std + mean
            _x_vhr = _x_vhr * std + mean
            _x_vhr_rec = _x_vhr_rec * std + mean

            save_image(_x_vhr, file_name_input)
            save_image(_x_vhr_rec, file_name_output)
            save_image(_x_vhr_both, file_name_both)


def main():
    args, config = parse_option()

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(config.LOCAL_RANK)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    wandb_logger = None

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

        with open(args.cfg, 'r') as f:
            yaml_config = yaml.load(f, Loader=yaml.FullLoader)

        wandb_logger = wandb.init(
            project=config.MODEL.NAME,
            dir=config.OUTPUT,
            config=yaml_config
        )

    # print config
    logger.info(config.dump())

    try:
        train(config, logger, wandb_logger)
        exit_code = 0
    except:
        exit_code = 1

    wandb.finish(exit_code)
    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
