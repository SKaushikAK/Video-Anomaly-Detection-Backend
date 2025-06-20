import torch
import torch.nn as nn
import cv2
import numpy as np
from transformers import SwinForImageClassification, AutoImageProcessor

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define Swin Transformer model (Same as training)
class SwinVideoClassifier(nn.Module):
    def __init__(self, model_name="microsoft/swin-base-patch4-window7-224"):
        super(SwinVideoClassifier, self).__init__()
        self.swin = SwinForImageClassification.from_pretrained(
            model_name, num_labels=2, ignore_mismatched_sizes=True
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.swin(x).logits
        x = x.view(B, T, -1)
        x = torch.mean(x, dim=1)
        return x

# Load trained model
model_path = "utils/swin_fight_detection.pth"
model = SwinVideoClassifier().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Load image processor
processor = AutoImageProcessor.from_pretrained("microsoft/swin-base-patch4-window7-224")

# Function to extract frames from video
def extract_frames(video_path, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224))
        frames.append(frame)

    cap.release()

    # Ensure exactly 16 frames (duplicate last frame if needed)
    while len(frames) < num_frames:
        frames.append(frames[-1])

    return frames

def predict_video(video_path):
    frames = extract_frames(video_path)
    inputs = processor(images=frames, return_tensors="pt")["pixel_values"]
    inputs = inputs.squeeze(0).to(device)  # Shape: (T, C, H, W)

    inputs = inputs.unsqueeze(0)  # Now shape: (1, T, C, H, W)

    with torch.no_grad():
        outputs = model(inputs)
        prediction = torch.argmax(outputs).item()

    return "Fight" if prediction == 1 else "NonFight"


if __name__ == "__main__":
    # Run prediction
    video_path = "D:/Anomaly/IT HOD/IT HOD/MAINGATE IN -1.mp4"  # Change this to your test video path
    result = predict_video(video_path)
    print(f"Prediction: {result}")
