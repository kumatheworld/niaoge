import os
import random
import numpy as np
import yaml
import torch
import torch.nn as nn

# dirty hack to get PANNs to work without changing code
import sys
sys.path.append('audioset_tagging_cnn/pytorch')
import audioset_tagging_cnn.pytorch.models

def fix_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def prepare_model(train, Model, ckpt_path, train_from, device):
    # load a PANN model
    num_classes_audioset = 527
    num_birds = 264

    ckpt = torch.load(ckpt_path, map_location=device)
    model = Model(sample_rate=32000, window_size=1024,
                  hop_size=320, mel_bins=64, fmin=50, fmax=14000,
                  classes_num=num_classes_audioset if train else num_birds)
    model.load_state_dict(ckpt['model'])

    if train:
        hidden_size = 2048
        # replace the final layer
        if Model.__name__ == 'Cnn14_DecisionLevelAtt':
            model.att_block = audioset_tagging_cnn.pytorch.models.AttBlock(
                hidden_size, num_birds, activation='sigmoid')
        else:
            model.fc_audioset = nn.Linear(hidden_size, num_birds)

        for name, param in model.named_parameters():
            if train_from in name:
                break
            param.requires_grad = False

    model.to(device)

    return model

def set_config(config_name, train):
    '''
    Load config and replace some string values with corresponding objects
    including device and model.
    '''

    config_dir = 'configs'
    config_path = os.path.join(config_dir, config_name + '.yaml')
    with open(config_path) as f:
        cfg = yaml.load(f, yaml.SafeLoader)

    if train and cfg['SEED']:
        fix_seed(cfg['SEED'])

    use_cuda = torch.cuda.is_available() and cfg['USE_CUDA']
    device = torch.device('cuda' if use_cuda else 'cpu')
    cfg['USE_CUDA'] = use_cuda
    cfg['DEVICE'] = device

    Model = getattr(audioset_tagging_cnn.pytorch.models, cfg['MODEL'])
    ckpt_path = cfg['PRETRAINED_PATH'] if train else cfg['CKPT_PATH']
    model = prepare_model(train, Model, ckpt_path, cfg['TRAIN_FROM'], device)
    cfg['MODEL'] = model

    return cfg
