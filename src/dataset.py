import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
import config

class CSLDataset(Dataset):
    def __init__(self, split='train'):
        self.split = split
        with open(config.LABEL_MAP_FILE, 'r') as f:
            self.label_map = json.load(f)
        with open(config.SPLITS_FILE, 'r') as f:
            splits = json.load(f)
            self.file_paths = splits[split]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        rel_path = self.file_paths[idx]
        label_str = os.path.dirname(rel_path)
        label_id = self.label_map[label_str]
        
        full_path = os.path.join(config.KEYPOINTS_DIR, rel_path)
        frames = np.load(full_path)
        frames = np.nan_to_num(frames)
        
        seq_len, feature_dim = frames.shape
        
        # Padding / Truncating
        if seq_len < config.MAX_FRAMES:
            padding = np.zeros((config.MAX_FRAMES - seq_len, feature_dim))
            frames = np.vstack((frames, padding))
        elif seq_len > config.MAX_FRAMES:
            frames = frames[:config.MAX_FRAMES, :]
            
        frames_tensor = torch.FloatTensor(frames)
        
        # --- DATA AUGMENTATION ---
        # If training, add slight Gaussian noise to the keypoints (simulates variations in movement)
        if self.split == 'train':
            noise = torch.randn_like(frames_tensor) * 0.005 # 0.5% spatial noise
            frames_tensor = frames_tensor + noise

        label_tensor = torch.LongTensor([label_id]).squeeze()
        return frames_tensor, label_tensor