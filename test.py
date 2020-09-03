import argparse
from torch.utils.data import DataLoader
import torch
import numpy as np

from load import set_config
from dataset import TestDataset
from evaluate import binarize, mean_f1_score

def test(cfg):
    dataset = TestDataset()
    kwargs = {
        'num_workers': cfg['NUM_WORKERS'],
        'pin_memory': True
    } if cfg['USE_CUDA'] else {}
    loader = DataLoader(dataset, cfg['BATCH_SIZE'], kwargs)

    device = cfg['DEVICE']
    model = cfg['MODEL']

    model.eval()
    predictions = []
    with torch.no_grad():
        for data, label in loader:
            data = data.to(device)
            label = label.to(device)
            pred = model(data)['clipwise_output']
            predictions.append(pred.cpu().detach().numpy())

    pred = np.concatenate(predictions)
    label = dataset.label
    thresholds = np.linspace(0, 1, 101)
    for threshold in thresholds:
        pred_bin = binarize(pred, threshold)
        score = mean_f1_score(label, pred_bin)
        print(threshold, score)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='default',
                        help='YAML file name under config/')
    args = parser.parse_args()

    cfg = set_config(args.config, train=False)
    test(cfg)
