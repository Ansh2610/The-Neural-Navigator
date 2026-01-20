import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

from data_loader import create_dataloaders
from model import NeuralNavigator


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for batch in tqdm(loader, desc="Training", leave=False):
        images = batch["image"].to(device)
        texts = batch["text"].to(device)
        paths = batch["path"].to(device)
        
        optimizer.zero_grad()
        predictions = model(images, texts)
        loss = criterion(predictions, paths)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating", leave=False):
            images = batch["image"].to(device)
            texts = batch["text"].to(device)
            paths = batch["path"].to(device)
            
            predictions = model(images, texts)
            loss = criterion(predictions, paths)
            total_loss += loss.item()
    
    return total_loss / len(loader)


def plot_losses(train_losses, val_losses, save_path="loss_curves.png"):
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, "b-", label="Train Loss")
    plt.plot(epochs, val_losses, "r-", label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Loss Curves")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Loss curves saved to {save_path}")


def train(
    data_dir="data",
    batch_size=32,
    epochs=50,
    lr=1e-3,
    device=None,
    save_dir="checkpoints",
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(save_dir, exist_ok=True)
    
    train_loader, val_loader, vocab = create_dataloaders(data_dir, batch_size=batch_size)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    model = NeuralNavigator(vocab_size=len(vocab)).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
            }, os.path.join(save_dir, "best_model.pth"))
            print(f"Saved best model (val_loss: {val_loss:.6f})")
        
        if epoch % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
            }, os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pth"))
    
    plot_losses(train_losses, val_losses)
    
    torch.save({
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_losses[-1],
        "val_loss": val_losses[-1],
    }, os.path.join(save_dir, "final_model.pth"))
    
    print(f"\nTraining complete! Best val loss: {best_val_loss:.6f}")
    return model, train_losses, val_losses


if __name__ == "__main__":
    train(epochs=50, batch_size=32, lr=1e-3)
