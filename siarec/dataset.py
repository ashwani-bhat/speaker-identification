import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader



class ASRDataset(Dataset):
    def __init__(self, path, labels):
        self.path = path
        self.labels = labels

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        pass

