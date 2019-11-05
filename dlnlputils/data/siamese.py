import torch
from torch.utils.data import Dataset
import random

from .nnets import ensure_length


class PairDataset(Dataset):
    def __init__(self, left_texts, right_texts, targets, out_len, pad_value=0):
        self.left_texts = left_texts
        self.right_texts = right_texts
        self.targets = targets
        self.out_len = out_len
        self.pad_value = pad_value

    def __len__(self):
        return len(self.left_texts)

    def __getitem__(self, item):
        left_text = ensure_length(self.left_texts[item], self.out_len, self.pad_value)
        right_text = ensure_length(self.right_texts[item], self.out_len, self.pad_value)

        if random.random() < 0.5:
            left_text, right_text = right_text, left_text

        inputs = torch.stack((torch.tensor(left_text, dtype=torch.long),
                              torch.tensor(right_text, dtype=torch.long)),
                             dim=0)
        cur_target = torch.tensor(self.targets[item], dtype=torch.float)

        return inputs, cur_target
