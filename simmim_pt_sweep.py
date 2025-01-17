import os
import argparse
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from config import get_config
from logger import create_logger
from main_simmim_pt import train

import wandb

# pytorch major version (1.x or 2.x)
PYTORCH_MAJOR_VERSION = int(torch.__version__.split('.')[0])


RESUME_SWEEP = None # 'sweep_id'

SWEEP_CONFIG = {
    'program': 'main_simmim_pt.py',
    'name': 'simmim_pretrain_sweep',
    'method': 'bayes',
    'metric': {
        'name': 'loss',
        'goal': 'minimize'
    },
    'parameters': {
        'MODEL': {
            'parameters': {
                'TYPE': {
                    'value': 'swinv2'
                },
                'NAME': {
                    'value': 'simmim_pretrain'
                },
                'DROP_PATH_RATE': {
                    'distribution': 'uniform',
                    'min': 0.0, 'max': 0.5
                },
                'SIMMIM': {
                    'parameters': {
                        'NORM_TARGET': {
                            'parameters': {
                                'ENABLE': {
                                    'value': True
                                },
                                'PATCH_SIZE': {
                                    # 'distribution': 'int_uniform',
                                    # 'min': 24, 'max': 94
                                    'values': [23, 47, 71]
                                }
                            }
                        }
                    }
                },
                'SWINV2': {
                    'parameters': {
                        'EMBED_DIM': {
                            # 'distribution': 'q_uniform',
                            # 'min': 64, 'max': 256, 'q': 32
                            'value': 128 
                        },
                        'DEPTHS': {
                            'value': [2, 2, 18, 2]
                        },
                        'NUM_HEADS': {
                            'value': [4, 8, 16, 32]
                        },
                        'WINDOW_SIZE': {
                            'values': [8, 16]
                        }
                    }
                }
            }
        },
        'DATA': {
            'parameters': {
                'BATCH_SIZE': {
                    'distribution': 'q_uniform',
                    # 'min': 8, 'max': 64, 'q': 8
                    'min': 8, 'max': 32, 'q': 8
                },
                'IMG_SIZE': {
                    'value': 256
                },
                'MASK_PATCH_SIZE': {
                    'distribution': 'q_uniform',
                    'min': 16, 'max': 32, 'q': 16
                },
                'MASK_RATIO': {
                    'distribution': 'q_uniform',
                    'min': 0.1, 'max': 0.9, 'q': 0.1
                }
            }
        },
        'TRAIN': {
            'parameters': {
                'EPOCHS': {
                    'value': 800
                },
                'WARMUP_EPOCHS': {
                    'value': 10
                },
                'BASE_LR': {
                    'values': [1e-1, 1e-2, 1e-3, 1e-4]
                },
                'WARMUP_LR': {
                    'value': 1e-5
                },
                'WEIGHT_DECAY': {
                    'distribution': 'uniform',
                    'min': 0.0, 'max': 0.2
                },
                'LR_SCHEDULER': {
                    'parameters': {
                        'NAME': {
                            'value': 'multistep'
                        },
                        'GAMMA': {
                            'values': [0.1, 0.25, 0.5]
                        },
                        'MULTISTEPS': {
                            'value': [200, 400, 600]
                        }
                    }
                }
            }
        },
        'PRINT_FREQ': {
            'value': 100
        },
        'SAVE_FREQ': {
            'value': -1
        },
        'TAG': {
            'value': 'simmim_pretrain__swinv2_base'
        }
    }
}


def parse_option():
    parser = argparse.ArgumentParser('SimMIM pre-training script', add_help=False)
    # parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
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
    parser.add_argument('--tag', help='tag of experiment')

    # distributed training
    # for pytorch >= 2.0, use `os.environ['LOCAL_RANK']` instead
    # (see https://pytorch.org/docs/stable/distributed.html#launch-utility)
    if PYTORCH_MAJOR_VERSION == 1:
        parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')

    args = parser.parse_args()

    config = get_config(args, wandb_sweep=True)

    return args, config


def main(config=None):
    if int(os.environ['LOCAL_RANK']) == 0:
        wandb_logger = wandb.init(project=project)
    else:
        wandb_logger = None

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

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())

    train(config, logger, wandb_logger)


if __name__ == '__main__':
    project = SWEEP_CONFIG['parameters']['MODEL']['parameters']['NAME']['value']

    if RESUME_SWEEP is None:
        sweep_id = wandb.sweep(SWEEP_CONFIG, project=project)
    else:
        sweep_id = RESUME_SWEEP

    wandb.agent(sweep_id, main, project=project)
