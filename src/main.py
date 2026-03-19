import train as train
from helpers import check_device
import yaml

# does git work?
def run_training(config):
    train.train_model(config)
    print("Testing complete.")


# test function to you all!
def run_inference():
    print("Inference not yet implimented")


if __name__ == "__main__":
    device = check_device()
    with open("config/train.config.yaml") as f:
        config = yaml.safe_load(f)

    run_training(config)
    run_inference()  # Not implimented
