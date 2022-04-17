import random
from .dataset import ASRDataset
import numpy as np


def create_pairs(path, speaker):
    x0_data = []
    x1_data = []
    label = []

    for i in range(len(path)):
        for j in range(len(path)):
            if i != j:
                x0_data.append(path[i])
                x1_data.append(path[j])    
                if speaker[i] == speaker[j]:
                    # Positive sample
                    label.append(1)
                else:
                    # Negative sample
                    label.append(0)
                    
    return x0_data, x1_data, label


def create_iterator(path, speaker, batch_size, shuffle=False):
    x0, x1, speaker = create_pairs(path, speaker)
    ret = ASRDataset(x0, x1, speaker)
    return ret