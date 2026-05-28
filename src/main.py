import train as train
from utility.hardware import check_device

# does git work?


def run_training():
    train.train_model()
    print("Testing complete.")


# test function to you all!
def run_inference():
    print("Inference not yet implimented")


if __name__ == "__main__":
    device = check_device()
    run_training()
    run_inference()  # Not implimented
