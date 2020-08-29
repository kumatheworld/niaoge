import argparse

from load import set_config

def test(cfg):
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='default',
                        help='YAML file name under config/')
    args = parser.parse_args()

    cfg = set_config(args.config, train=False)
    test(cfg)
