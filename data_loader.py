import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np


class Vocabulary:
    def __init__(self):
        self.word2idx = {
            "<pad>": 0,
            "go": 1,
            "to": 2,
            "the": 3,
            "red": 4,
            "blue": 5,
            "green": 6,
            "circle": 7,
            "square": 8,
            "triangle": 9,
        }
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.pad_idx = 0

    def encode(self, text, max_len=6):
        words = text.lower().split()
        indices = [self.word2idx.get(w, self.pad_idx) for w in words]
        if len(indices) < max_len:
            indices += [self.pad_idx] * (max_len - len(indices))
        return indices[:max_len]

    def __len__(self):
        return len(self.word2idx)


class NavigatorDataset(Dataset):
    def __init__(self, data_dir, is_test=False):
        self.data_dir = data_dir
        self.is_test = is_test
        self.image_dir = os.path.join(data_dir, "images")
        self.annotation_dir = os.path.join(data_dir, "annotations")
        self.vocab = Vocabulary()
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        annotation_files = sorted(os.listdir(self.annotation_dir))
        for ann_file in annotation_files:
            if ann_file.endswith(".json"):
                with open(os.path.join(self.annotation_dir, ann_file)) as f:
                    samples.append(json.load(f))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        img_path = os.path.join(self.image_dir, sample["image_file"])
        image = Image.open(img_path).convert("RGB")
        image = np.array(image, dtype=np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        
        text_indices = self.vocab.encode(sample["text"])
        text_tensor = torch.tensor(text_indices, dtype=torch.long)
        
        if self.is_test:
            return {
                "image": image,
                "text": text_tensor,
                "id": sample["id"],
                "image_file": sample["image_file"],
                "text_str": sample["text"],
            }
        
        path = np.array(sample["path"], dtype=np.float32) / 128.0
        path_tensor = torch.from_numpy(path)
        
        return {
            "image": image,
            "text": text_tensor,
            "path": path_tensor,
            "id": sample["id"],
        }


def create_dataloaders(data_dir, batch_size=32, val_split=0.1, num_workers=0):
    dataset = NavigatorDataset(data_dir, is_test=False)
    
    total_size = len(dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    return train_loader, val_loader, dataset.vocab


def create_test_dataloader(test_dir, batch_size=32, num_workers=0):
    dataset = NavigatorDataset(test_dir, is_test=True)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return loader, dataset.vocab


if __name__ == "__main__":
    train_loader, val_loader, vocab = create_dataloaders("data", batch_size=4)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Vocab size: {len(vocab)}")
    
    batch = next(iter(train_loader))
    print(f"\nSample batch:")
    print(f"  Image shape: {batch['image'].shape}")
    print(f"  Text shape: {batch['text'].shape}")
    print(f"  Path shape: {batch['path'].shape}")
    print(f"  Path range: [{batch['path'].min():.3f}, {batch['path'].max():.3f}]")
