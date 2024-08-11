import torch as T

class CustomDataset(T.utils.data.Dataset):
    """
    data: iterable
    """
    def __init__(self, data, device):
        self.data = T.tensor(data).float().to(device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]