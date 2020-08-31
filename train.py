import argparse
import os
import torch
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from evaluate import binarize, mean_f1_score
from load import set_config
from dataset import BirdcallDataset
import losses

def get_loader(df, cfg):
    dataset =  BirdcallDataset(df, cfg['LIKELIHOOD'], cfg['AUDIO_DURATION'])
    kwargs = {
        'num_workers': cfg['NUM_WORKERS'],
        'pin_memory': True
    } if cfg['USE_CUDA'] else {}
    loader = DataLoader(dataset, cfg['BATCH_SIZE'], kwargs)
    return loader

def train(cfg):
    device = cfg['DEVICE']
    model = cfg['MODEL']

    # prepare loaders
    train_csv = os.path.join('data', 'train.csv')
    df = pd.read_csv(train_csv)

    if cfg['SANITY_CHECK']:
        df = df.drop_duplicates(subset=['ebird_code'])

    if cfg['VALIDATION']:
        train_df, test_df = train_test_split(df)
        train_loader = get_loader(train_df, cfg)
        val_loader = get_loader(test_df, cfg)
    else:
        train_loader = get_loader(df, cfg)
        val_loader = None

    # prepare some more
    criterion = getattr(losses, cfg['LOSS'])
    optimizer = getattr(optim, cfg['OPTIM_ALGO'])(
        [param for param in model.parameters() if param.requires_grad],
        lr=cfg['LR']
    )
    ckpt_dir = os.path.dirname(cfg['CKPT_PATH'])
    os.makedirs(ckpt_dir, exist_ok=True)
    writer = SummaryWriter()

    # start training!
    n_iter = 1
    best_score = 0
    for epoch in range(1, cfg['EPOCHS'] + 1):
        # train
        model.train()
        train_loss = 0
        train_score = 0
        for data, label in train_loader:
            optimizer.zero_grad()
            data = data.to(device)
            label = label.to(device)

            pred = model(data)['clipwise_output']
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()

            loss_item = loss.item()
            pred_bin = binarize(pred, cfg['PRED_THRESH'])
            score = mean_f1_score(label, pred_bin)

            writer.add_scalar('loss/train', loss_item, n_iter)
            writer.add_scalar('score/train', score, n_iter)
            train_loss += loss_item
            train_score += score

            if cfg['SANITY_CHECK']:
                print(n_iter, loss_item, score)

            n_iter += 1

        train_loss /= len(train_loader)
        train_score /= len(train_loader)

        # validate
        val_score = 0
        val_loss = 0
        if cfg['VALIDATION']:
            model.eval()
            with torch.no_grad():
                for data, label in val_loader:
                    data = data.to(device)
                    label = label.to(device)

                    pred = model(data)['clipwise_output']
                    val_loss += criterion(pred, label).item()
                    pred_bin = binarize(pred, cfg['PRED_THRESH'])
                    val_score += mean_f1_score(label, pred_bin)

            val_loss /= len(val_loader)
            val_score /= len(val_loader)
            writer.add_scalars('loss/train & val',
                            {'train': train_loss, 'val': val_loss}, epoch)
            writer.add_scalars('score/train & val',
                            {'train': train_score, 'val': val_score}, epoch)

        # save model
        if best_score <= val_score and not cfg['SANITY_CHECK']:
            best_score = val_score
            checkpoint = {
                'config': str(cfg),
                'epoch': epoch,
                'model': model.state_dict()
            }
            torch.save(checkpoint, cfg['CKPT_PATH'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='default',
                        help='YAML file name under config/')
    args = parser.parse_args()

    cfg = set_config(args.config, train=True)
    train(cfg)
