import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
import torch
from skimage import io
import numpy as np

class ASRDataset(Dataset):
    def __init__(self, x0, x1, speaker):
        self.x0 = x0
        self.x1 = x1
        self.speaker = speaker

    def __len__(self):
        return len(self.speaker)
    
    def __getitem__(self, idx):
        image0 = io.imread(self.x0[idx])
        image1 = io.imread(self.x1[idx])
        image0 = image0/255.0 # normalize data
        image1 = image1/255.0 # normalize data

        labels = torch.tensor(self.speaker[idx], dtype=torch.float)
        return (image0, image1, labels)


class ASRTestDataset(Dataset):
    def __init__(self, x0, speaker):
        self.x0 = x0
        self.speaker = speaker

    def __len__(self):
        return len(self.speaker)
    
    def __getitem__(self, idx):
        image0 = io.imread(self.x0[idx])
        
        labels = torch.tensor(self.speaker[idx], dtype=torch.float)
        return (image0, labels)
