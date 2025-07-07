import yaml
import argparse
from train.train_classification import train as train_classifier  # ✅

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/default.yaml')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if cfg['model']['type'] == 'classifier':
        train_classifier(cfg)
    else:
        raise NotImplementedError("Only classification supported now")
