import wandb


PROJECT = 'simmim_pretrain'

SWEEP_CONFIG = {
    'program': 'main_simmim_pt.py',
    'name': f'{PROJECT}_sweep',
    'method': 'bayes',
    'metric': {
        'name': 'val_loss',
        'goal': 'minimize'
    },
    'parameters': {
        'MODEL': {
            'parameters': {
                'TYPE': {
                    'value': 'swinv2'
                },
                'NAME': {
                    'value': PROJECT
                },
                'DROP_PATH_RATE': {
                    'distribution': 'uniform',
                    'min': 0.0, 'max': 0.5
                },
                'SWINV2': {
                    'parameters': {
                        'EMBED_DIM': {
                            'value': 128
                        },
                        'DEPTHS': {
                            'value': [2, 2, 18, 2]
                        },
                        'NUM_HEADS': {
                            'value': [4, 8, 16, 32]
                        },
                        'WINDOW_SIZE': {
                            'value': 12
                        }
                    }
                }
            }
        },
        'DATA': {
            'parameters': {
                'BATCH_SIZE': {
                    'value': 4
                },
                'IMG_SIZE': {
                    'value': 192
                },
                'MASK_PATCH_SIZE': {
                    'value': 4 # 'values': [4, 8, 16, 32, 64]
                },
                'MASK_RATIO': {
                    'value': 0.5
                },
                'NUM_WORKERS': {
                    'value': 16
                }
            }
        },
        'TRAIN': {
            'parameters': {
                'EPOCHS': {
                    'value': 400
                },
                'WARMUP_EPOCHS': {
                    'value': 10
                },
                'BASE_LR': {
                    'values': [1e-4, 1e-3, 1e-2, 1e-1]
                },
                'MIN_LR': {
                    'value': 1e-6
                },
                'WARMUP_LR': {
                    'value': 1e-6
                },
                'WEIGHT_DECAY': {
                    'distribution': 'uniform',
                    'min': 0.0, 'max': 0.5
                },
                'LR_SCHEDULER': {
                    'parameters': {
                        'NAME': {
                            'value': 'plateau'
                        },
                        'PATIENCE_T': {
                            'values': [10, 20]
                        },
                        'THRESHOLD': {
                            'value': 1e-2
                        },
                        'COOLDOWN_T': {
                            'values': [0, 10]
                        },
                        'MODE': {
                            'value': 'min'
                        },
                        'EARLY_STOP': {
                            'value': True
                        }
                    }
                },
                'OPTIMIZER': {
                    'parameters': {
                        'NAME': {
                            'value': 'adamw'
                        },
                        'BETA_1': {
                            'distribution': 'uniform',
                            'min': 0.85, 'max': 0.95
                        },
                        'BETA_2': {
                            'distribution': 'uniform',
                            'min': 0.9, 'max': 0.999
                        }
                    }
                },
                'ACCUMULATION_STEPS': {
                    'values': [1, 4, 8]
                }
            }
        },
        'PRINT_FREQ': {
            'value': 20
        },
        'SAVE_FREQ': {
            'value': -1
        },
        'TAG': {
            'value': 'simmim_pretrain__swinv2_base'
        }
    }
}


if __name__ == '__main__':
    sweep_id = wandb.sweep(SWEEP_CONFIG, project=PROJECT)
