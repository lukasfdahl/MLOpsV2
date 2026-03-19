from config import config
import os


def test_dataset_availability():
    if config["settings"]["use_sample_dataset"]:
        assert os.path.exists(config["path"]["sample_dataset_path"]), f"Sample dataset is set used and the files exist at {config['path']['sample_dataset_path']}"
    else:
        assert os.path.exists(config["path"]["full_dataset_path"]), f"Full dataset is set used and the files exist at {config['path']['full_dataset_path']}"
