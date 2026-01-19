import torch
import torch.nn as nn


class VisionEncoder(nn.Module):
    def __init__(self, output_dim=256):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, output_dim)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class TextEncoder(nn.Module):
    def __init__(self, vocab_size=10, embed_dim=32, output_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc = nn.Linear(embed_dim * 6, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    vision_enc = VisionEncoder()
    text_enc = TextEncoder()
    
    dummy_image = torch.randn(4, 3, 128, 128)
    dummy_text = torch.randint(0, 10, (4, 6))
    
    vision_out = vision_enc(dummy_image)
    text_out = text_enc(dummy_text)
    
    print(f"Vision encoder output: {vision_out.shape}")
    print(f"Text encoder output: {text_out.shape}")
    print(f"Combined would be: {vision_out.shape[1] + text_out.shape[1]} dims")
