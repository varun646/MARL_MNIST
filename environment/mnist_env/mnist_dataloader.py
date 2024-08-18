from torchvision.datasets
from torch.utils.data import Dataset

# NOTE: MAY NEED TO REMOVE THIS IF MNIST DATASET CAN EASILY BE INTEGRATED IN MARL_MNIST.PY FILE


class MNISTMasked(Dataset):
    def __init__(self):
        # TODO: initialize with masked images
        self.images = []

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]
