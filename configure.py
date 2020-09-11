import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import oyaml as yaml
from dotmap import DotMap

import losses

# dirty hack to get PANNs to work without changing code
import sys
sys.path.append('audioset_tagging_cnn/pytorch')
import audioset_tagging_cnn.pytorch.models

def fix_seed(seed):
    os.environ.PYTHONHASHSEED = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def prepare_model(Model, state_dict, device, train=False, train_from=None):
    # load a PANN model
    classes_num = [
        b.size(0) for name, b in state_dict.items() if name.endswith('.bias')
    ][-1]
    model = Model(sample_rate=32000, window_size=1024, hop_size=320,
                  mel_bins=64, fmin=50, fmax=14000, classes_num=classes_num)
    model.load_state_dict(state_dict)

    if train:
        num_birds = 264
        if classes_num != num_birds:
            # replace the final layer
            hidden_size = 2048
            if Model.__name__ == 'Cnn14_DecisionLevelAtt':
                model.att_block = audioset_tagging_cnn.pytorch.models.AttBlock(
                    hidden_size, num_birds, activation='sigmoid')
            else:
                model.fc_audioset = nn.Linear(hidden_size, num_birds)

        # freeze first few layers
        for name, param in model.named_parameters():
            if name.startswith(train_from):
                break
            param.requires_grad = False
        else:
            raise Exception(f'layer {train_from} not found')

    model.to(device)

    return model

class Config():
    def __init__(self, config_name, train):
        self.name = config_name
        config_dir = 'configs'
        config_path = os.path.join(config_dir, config_name + '.yaml')

        with open(config_path) as f:
            self.str = f.read()
        self.cfg = yaml.load(self.str, yaml.SafeLoader)

        dm = DotMap(self.cfg)
        self.__dict__.update(dm.__dict__['_map'])

        if train and self.SEED is not None:
            fix_seed(self.SEED)

        use_cuda = torch.cuda.is_available() and self.USE_CUDA
        device = torch.device('cuda' if use_cuda else 'cpu')
        self.USE_CUDA = use_cuda
        self.DEVICE = device

        Model = getattr(audioset_tagging_cnn.pytorch.models, self.MODEL)
        halftrained_path = self.HALFTRAINED_PATH
        ckpt_path = (halftrained_path if halftrained_path else self.PRETRAINED_PATH) \
                    if train else self.CKPT_PATH
        ckpt = torch.load(ckpt_path, map_location=device)
        state_dict = ckpt['model']
        train_from = self.TRAIN_FROM
        model = prepare_model(Model, state_dict, device, train, train_from)
        self.MODEL = model

        if train:
            self.LOSS = getattr(losses, self.LOSS)()

            optimizer = getattr(optim, self.OPTIMIZER)(
                [param for param in model.parameters() if param.requires_grad],
                lr=self.LR.LR
            )
            if self.RESUME:
                optimizer.load_state_dict(ckpt['optimizer'])
            self.OPTIMIZER = optimizer

            scheduler = getattr(optim.lr_scheduler, self.LR.SCHEDULER)(
                optimizer,
                **self.LR.KWARGS
            )
            if self.RESUME:
                scheduler.load_state_dict(ckpt['scheduler'])
            self.LR.SCHEDULER = scheduler

    def get_markdown_string(self):
        return f'<pre>{self.str}</pre>'
