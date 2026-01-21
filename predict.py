import os
import torch
import cv2
import numpy as np
from tqdm import tqdm

from data_loader import create_test_dataloader, Vocabulary
from model import NeuralNavigator


def load_model(checkpoint_path, device):
    model = NeuralNavigator()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def draw_path(image, path, color=(0, 0, 255), thickness=2, radius=4):
    img = (image * 255).astype(np.uint8).copy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    points = (path * 128).astype(np.int32)
    
    for i in range(len(points) - 1):
        pt1 = tuple(points[i])
        pt2 = tuple(points[i + 1])
        cv2.line(img, pt1, pt2, color, thickness)
    
    for i, (x, y) in enumerate(points):
        if i == 0:
            cv2.circle(img, (x, y), radius + 1, (0, 255, 0), -1)
        elif i == len(points) - 1:
            cv2.circle(img, (x, y), radius + 1, (255, 0, 0), -1)
        else:
            cv2.circle(img, (x, y), radius, color, -1)
    
    return img


def predict_and_save(model, test_loader, output_dir, device):
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            images = batch["image"].to(device)
            texts = batch["text"].to(device)
            image_files = batch["image_file"]
            text_strs = batch["text_str"]
            
            predictions = model(images, texts)
            predictions = predictions.cpu().numpy()
            images_np = images.cpu().numpy().transpose(0, 2, 3, 1)
            
            for i in range(len(predictions)):
                img_with_path = draw_path(images_np[i], predictions[i])
                
                text_label = text_strs[i]
                cv2.putText(
                    img_with_path, text_label, (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1
                )
                
                output_path = os.path.join(output_dir, f"pred_{image_files[i]}")
                cv2.imwrite(output_path, img_with_path)


def predict_single(model, image_path, text, vocab, device):
    from PIL import Image
    
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image, dtype=np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).to(device)
    
    text_indices = vocab.encode(text)
    text_tensor = torch.tensor(text_indices, dtype=torch.long).unsqueeze(0).to(device)
    
    with torch.no_grad():
        prediction = model(image_tensor, text_tensor)
    
    path = prediction[0].cpu().numpy()
    img_with_path = draw_path(image_np, path)
    
    return img_with_path, path


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    checkpoint_path = "checkpoints/best_model.pth"
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}")
        print("Please run train.py first to train the model.")
        exit(1)
    
    model = load_model(checkpoint_path, device)
    print("Model loaded successfully")
    
    test_loader, vocab = create_test_dataloader("test_data", batch_size=16)
    print(f"Test samples: {len(test_loader.dataset)}")
    
    predict_and_save(model, test_loader, "predictions", device)
    print("Predictions saved to predictions/")
