import torch
import torch.nn as nn
import math


class VisionEncoder(nn.Module):
    def __init__(self, output_dim=256):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
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


class TransformerPathDecoder(nn.Module):
    def __init__(self, input_dim=320, hidden_dim=128, num_points=10, num_heads=4, num_layers=2):
        super().__init__()
        self.num_points = num_points
        self.hidden_dim = hidden_dim
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        self.position_embed = nn.Embedding(num_points, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_proj = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        batch_size = x.size(0)
        
        context = self.input_proj(x)
        
        positions = torch.arange(self.num_points, device=x.device)
        pos_embed = self.position_embed(positions)
        
        queries = pos_embed.unsqueeze(0).expand(batch_size, -1, -1)
        queries = queries + context.unsqueeze(1)
        
        out = self.transformer(queries)
        
        coords = self.output_proj(out)
        coords = torch.sigmoid(coords)
        
        return coords


class NeuralNavigator(nn.Module):
    def __init__(self, vocab_size=10, vision_dim=256, text_dim=64, hidden_dim=128, num_points=10):
        super().__init__()
        self.vision_encoder = VisionEncoder(output_dim=vision_dim)
        self.text_encoder = TextEncoder(vocab_size=vocab_size, output_dim=text_dim)
        self.decoder = TransformerPathDecoder(
            input_dim=vision_dim + text_dim,
            hidden_dim=hidden_dim,
            num_points=num_points
        )

    def forward(self, image, text):
        vision_features = self.vision_encoder(image)
        text_features = self.text_encoder(text)
        combined = torch.cat([vision_features, text_features], dim=1)
        path = self.decoder(combined)
        return path


if __name__ == "__main__":
    model = NeuralNavigator()
    
    batch_size = 4
    dummy_image = torch.randn(batch_size, 3, 128, 128)
    dummy_text = torch.randint(0, 10, (batch_size, 6))
    
    output = model(dummy_image, dummy_text)
    print(f"Model output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
