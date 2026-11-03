"""Long training clips assembled on the fly from the 8 s training slices (build_long_clips.py).

Slice <song>_slice<k> covers frames 30k .. 30k + 240 of the song, and consecutive slices agree where they overlap,
so frame t of a clip that starts at slice k is frame t - 30 j of slice k + j, j = min(t // 30, n - 1). Taking each
frame from the earliest slice that holds it avoids each slice's last frame, whose foot-contact label is computed
without a next frame.
"""
import numpy as np
import torch
from torch.utils.data import Dataset

STEP = 30


class LongClipDataset(Dataset):
    def __init__(self, base, path):
        z = np.load(path)
        self.pose = base.data['pose']  # (N_slices, 240, 151), normalized
        self.normalizer = base.normalizer
        self.slices = torch.from_numpy(z['slices'].astype(np.int64))  # (N, n)
        self.features = z['features']
        self.genres = z['genres']
        self.names = list(z['names'])
        T = self.features.shape[1]
        j = torch.clamp(torch.arange(T) // STEP, max=self.slices.shape[1] - 1)
        self.j, self.local = j, torch.arange(T) - STEP * j
        assert (self.local < self.pose.shape[1]).all()

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        pose = self.pose[self.slices[i][self.j], self.local]
        return pose, self.features[i], self.names[i], self.names[i] + '.wav', self.genres[i]
