from torch.utils.data import Dataset
import torch

class FaceDataSet(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = torch.from_numpy(self.images[idx]).float(), self.labels[idx]
        return sample