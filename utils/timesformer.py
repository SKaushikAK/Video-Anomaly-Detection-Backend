import cv2
import torch
import numpy as np
from torchvision import transforms
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


# === TimeSformer Model Definition ===
class CustomTimeSformer(nn.Module):
    def __init__(self, num_classes=2, img_size=224, patch_size=16, num_frames=16):
        super(CustomTimeSformer, self).__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 768))
        self.pos_embed = nn.Parameter(torch.zeros(1, (img_size // patch_size) ** 2 + 1, 768))
        self.patch_embed = nn.Conv2d(3, 768, kernel_size=patch_size, stride=patch_size)

        encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=12)
        self.blocks = nn.ModuleList([encoder_layer for _ in range(6)])
        self.norm = nn.LayerNorm(768)
        self.head = nn.Linear(768, num_classes)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = x.reshape(B, T * x.shape[1], -1)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        if x.size(1) != self.pos_embed.size(1):
            pos_embed_resized = F.interpolate(
                self.pos_embed.transpose(1, 2),
                size=x.size(1),
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
            x = x + pos_embed_resized
        else:
            x = x + self.pos_embed[:, :x.size(1), :]

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        cls_output = x[:, 0]
        logits = self.head(cls_output)
        logits = logits.unsqueeze(1).repeat(1, T, 1)
        return logits


timesformer_model = None

def initialize():
    global timesformer_model
    if timesformer_model is None:
        timesformer_model = CustomTimeSformer()
        state_dict = torch.load("utils/timesformer_binary1.pth", map_location=device)
        if 'model' in state_dict:
            state_dict = state_dict['model']
        timesformer_model.load_state_dict(state_dict, strict=False)
        timesformer_model.to(device).eval()


def predict_video(video_path):
    initialize()
    video_tensor = extract_and_preprocess_frames(video_path)
    with torch.no_grad():
        try:
            logits = timesformer_model(video_tensor)
            probs = F.softmax(logits.squeeze(0), dim=1)
            anomaly_probs = probs[:, 1]

            mean_prob = anomaly_probs.mean().item()
            predicted_class = 1 if mean_prob > 0.5 else 0

            anomaly_frame_path = ""
            if predicted_class == 1:
                max_prob = np.max(anomaly_probs.cpu().numpy())
                anomaly_index = int(np.argmax(anomaly_probs.cpu().numpy()))
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

if __name__ == "__main__":
    video_path = r"d:\Machine learning\ML and Pattern Recognition\RWF-2000\val\NonFight\1MVS2QPWbHc_0.avi"
    result = predict_video(video_path)
    print(result)
