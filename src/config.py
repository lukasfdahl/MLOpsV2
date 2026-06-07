import yaml
import os

config_path = os.environ.get("TRAIN_CONFIG", "config/small_train.config.yaml")

with open(config_path) as f:
    config = yaml.safe_load(f)
