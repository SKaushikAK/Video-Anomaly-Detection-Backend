import cv2
import torch
import numpy as np
from torchvision import transforms
from transformers import ViTModel, ViTConfig, SwinForImageClassification
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
    transforms.Normalize([0.5]*3, [0.5]*3)  # Normalize to [-1, 1]
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

    video_tensor = torch.stack(frames)  # [T, 3, 224, 224]
    video_tensor = video_tensor.unsqueeze(0)  # [1, T, 3, 224, 224]
    return video_tensor.to(device)


class CustomViT(nn.Module):
    def __init__(self):
        super(CustomViT, self).__init__()
        config = ViTConfig.from_pretrained("google/vit-base-patch16-224-in21k")
        config.use_pooler = False
        self.vit = ViTModel(config)
        self.classifier = nn.Linear(config.hidden_size, 2)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        outputs = self.vit(x)
        cls_token = outputs.last_hidden_state[:, 0]
        logits = self.classifier(cls_token)
        logits = logits.view(B, T, -1)
        return torch.mean(logits, dim=1)


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
        return self.head(cls_output)


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
        x = self.swin(pixel_values=x).logits
        x = x.view(B, T, -1)
        return torch.mean(x, dim=1)



# Load models once and reuse them
vit_model, timesformer_model, swin_model = None, None, None

def model_initialize():
    global vit_model, timesformer_model, swin_model

    if vit_model is None:
        vit_model = CustomViT()
        vit_model.load_state_dict(torch.load("utils/vit_model.pth", map_location=device), strict=False)
        vit_model.vit.pooler = None
        vit_model.to(device).eval()

    if timesformer_model is None:
        timesformer_model = CustomTimeSformer()
        state_dict = torch.load("utils/timesformer_binary_final1.pth", map_location=device)
        if 'model' in state_dict:
            state_dict = state_dict['model']
        timesformer_model.load_state_dict(state_dict, strict=False)
        timesformer_model.to(device).eval()

    if swin_model is None:
        swin_model = SwinVideoClassifier()
        swin_model.load_state_dict(torch.load("utils/swin_fight_detection.pth", map_location=device))
        swin_model.to(device).eval()

    return vit_model, timesformer_model, swin_model


def predict_video(video_path):
    vit_model, timesformer_model, swin_model = model_initialize()
    video_tensor = extract_and_preprocess_frames(video_path)

    with torch.no_grad():
        try:
            vit_output = vit_model(video_tensor)
            timesformer_output = timesformer_model(video_tensor)
            swin_output = swin_model(video_tensor)

            vit_probs = F.softmax(vit_output, dim=1)
            timesformer_probs = F.softmax(timesformer_output, dim=1)
            swin_probs = F.softmax(swin_output, dim=1)

            ensemble_probs = (vit_probs + timesformer_probs + swin_probs) / 3
            
            predicted_class = torch.argmax(ensemble_probs, dim=1).item()

            return {
                "predicted_class": predicted_class,
                "label": "Anomaly (Fight)" if predicted_class == 1 else "Normal (No Fight)",
                "confidence": ensemble_probs.tolist()
            }

        except Exception as e:
            print("Error", e)
            return {"error": str(e)}
if __name__ == "__main__":
    # === Example Usage ===
    video_path = r"d:\Machine learning\ML and Pattern Recognition\RWF-2000\val\Fight\48J5lk4QcpE_2.avi"
    predict_video(video_path)

