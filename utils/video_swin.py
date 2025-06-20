import cv2
import torch
import numpy as np
from torchvision import transforms
from transformers import SwinForImageClassification
import torch.nn as nn
import torch.nn.functional as F

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preprocessing transformation
image_size = 224
num_frames = 16
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def extract_and_preprocess_frames(video_path, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idxs = np.linspace(0, frame_count - 1, num=num_frames, dtype=int)

    frames = []
    idx = 0
    ret = True
    while ret and len(frames) < num_frames:
        ret, frame = cap.read()
        if idx in frame_idxs:
            if frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                tensor = preprocess(frame)
                frames.append(tensor)
        idx += 1

    cap.release()

    if len(frames) < num_frames:
        while len(frames) < num_frames:
            frames.append(frames[-1])

    video_tensor = torch.stack(frames)
    video_tensor = video_tensor.unsqueeze(0)
    return video_tensor.to(device)


def get_original_frame_index(video_path, sampled_idx, total_samples=16):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idxs = np.linspace(0, total_frames - 1, num=total_samples, dtype=int)
    cap.release()
    return frame_idxs[sampled_idx]


def extract_frame_as_image(video_path, frame_number, save_path="anomalous_frame.jpg"):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    success, frame = cap.read()
    cap.release()

    if success:
        cv2.imwrite("uploads/" + save_path, frame)
        print(save_path)
        return save_path
    else:
        return None


# === Model Definitions ===
class SwinVideoClassifier(nn.Module):
    def __init__(self, model_name="microsoft/swin-base-patch4-window7-224", num_classes=2):
        super(SwinVideoClassifier, self).__init__()
        self.swin = SwinForImageClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        logits = self.swin(pixel_values=x).logits
        logits = logits.view(B, T, -1)
        return logits


# Load model once
swin_model = None

def model_initialize():
    global swin_model
    if swin_model is None:
        swin_model = SwinVideoClassifier()
        swin_model.load_state_dict(torch.load("utils/swin_fight_detection.pth", map_location=device))
        swin_model.to(device).eval()
    return swin_model


def initialize():
    global swin_model
    model_initialize()

def predict_video(video_path):
    model = model_initialize()
    video_tensor = extract_and_preprocess_frames(video_path)

    with torch.no_grad():
        try:
            logits = model(video_tensor)
            probs = F.softmax(logits.squeeze(0), dim=1)
            anomaly_probs = probs[:, 1]

            ensemble_probs_np = anomaly_probs.cpu().numpy()
            mean_prob = anomaly_probs.mean().item()
            predicted_class = 1 if mean_prob > 0.6 else 0
            
            anomaly_frame_path = ""
            if predicted_class == 1:
                anomaly_index = int(np.argmax(ensemble_probs_np))
                frame_number = get_original_frame_index(video_path, anomaly_index)
                anomaly_frame_path = extract_frame_as_image(video_path, frame_number, "anomalous_frame.jpg")

            return {
                "predicted_class": predicted_class,
                "label": "Anomaly (Fight)" if predicted_class == 1 else "Normal (No Fight)",
                "confidence": mean_prob,
                "route": "/uploads/",
                "anomalous_frame_path": anomaly_frame_path
            }

        except Exception as e:
            print("Error:", e)
            return {"error": str(e)}


# === Example usage ===
if __name__ == "__main__":
    video_path = r"d:\Machine learning\ML and Pattern Recognition\RWF-2000\val\Fight\0Ow4cotKOuw_4.avi"
    result = predict_video(video_path)
    print(result)
