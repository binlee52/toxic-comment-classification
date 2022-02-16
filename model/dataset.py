import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class KorHateDataset(Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, index):
        item = {key: torch.tensor(val[index]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[index])
        return item
    
    def __len__(self):
        return len(self.encodings["input_ids"])