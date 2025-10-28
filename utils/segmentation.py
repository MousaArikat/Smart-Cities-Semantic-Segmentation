import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torchvision.transforms as T
from torchvision import models
import numpy as np
from PIL import Image
import json
import os
import cv2
from moviepy.editor import ImageSequenceClip
from models.seg_hrnet import HighResolutionNet
from config import config as hrnet_config, update_config
import argparse
import torch.nn as nn

class HRNetWithDropout(nn.Module):
    def __init__(self, base_model, dropout_p=0.3):
        super(HRNetWithDropout, self).__init__()
        self.backbone = base_model
        in_channels = list(self.backbone.last_layer.modules())[-1].in_channels

        self.backbone.last_layer = nn.Identity()
        self.dropout = nn.Dropout2d(p=dropout_p)
        self.classifier = nn.Conv2d(in_channels, 19, kernel_size=1)

    def forward(self, x):
        x = self.backbone(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x



with open("assets/colors.json", "r") as f:
    color_map = json.load(f)
color_map = {int(k): tuple(v) for k, v in color_map.items()}

def load_model(weight_path, model_name):
    if model_name == "hrnetv2_w48":
        # Load HRNetV2-W48 with Cityscapes config
        args = argparse.Namespace(cfg="config/hrnet_config.yaml", opts=['DATASET.NUM_CLASSES', '19'])
        update_config(hrnet_config, args)
        model = HighResolutionNet(hrnet_config)

        # Replace classifier
        model.last_layer = torch.nn.Conv2d(in_channels=720, out_channels=19, kernel_size=1)

        # Load weights
        state_dict = torch.load(weight_path, map_location=torch.device("cpu"))
        model_state = model.state_dict()
        filtered_state_dict = {k: v for k, v in state_dict.items()
                               if k in model_state and v.shape == model_state[k].shape}
        model.load_state_dict(filtered_state_dict, strict=False)
        model.eval()
        return model
    
    elif model_name == "hrnetv2_w48_dropout":
        args = argparse.Namespace(cfg="config/hrnet_config.yaml", opts=['DATASET.NUM_CLASSES', '19'])
        update_config(hrnet_config, args)
        base_model = HighResolutionNet(hrnet_config)

        model = HRNetWithDropout(base_model, dropout_p=0.3)

        state_dict = torch.load(weight_path, map_location=torch.device("cpu"))
        model_state = model.state_dict()
        filtered_state_dict = {k: v for k, v in state_dict.items()
                            if k in model_state and v.shape == model_state[k].shape}
        model.load_state_dict(filtered_state_dict, strict=False)
        model.eval()
        return model


    elif "resnet101" in model_name:
        model = models.segmentation.deeplabv3_resnet101(pretrained=False, aux_loss=None)
        model.classifier[4] = torch.nn.Sequential(
            torch.nn.Dropout(0.3),
            torch.nn.Conv2d(256, 19, kernel_size=1)
        )
    

    elif model_name == "hrnetv2_w48_customhead":
        args = argparse.Namespace(cfg="config/hrnet_config.yaml", opts=['DATASET.NUM_CLASSES', '19'])
        update_config(hrnet_config, args)

        model = HighResolutionNet(hrnet_config)

        # Build custom classifier block just like training
        in_channels = list(model.last_layer.modules())[-1].in_channels
        model.last_layer = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Conv2d(256, 19, kernel_size=1)
        )

        # Load trained weights
        state_dict = torch.load(weight_path, map_location=torch.device("cpu"))
        model_state = model.state_dict()
        filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state and v.shape == model_state[k].shape}
        model.load_state_dict(filtered_state_dict, strict=False)

        model.eval()
        return model

    else:
        model = models.segmentation.deeplabv3_resnet50(pretrained=False, aux_loss=True)
        model.classifier[4] = torch.nn.Conv2d(256, 19, kernel_size=1)

    # Load weights for DeepLab models
    state_dict = torch.load(weight_path, map_location=torch.device('cpu'))
    model_state = model.state_dict()
    filtered_state_dict = {k: v for k, v in state_dict.items()
                           if k in model_state and v.shape == model_state[k].shape}
    model.load_state_dict(filtered_state_dict, strict=False)
    model.eval()
    return model



def preprocess(image):
    transform = T.Compose([
        T.Resize((512, 1024)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

def segment_image(model, image):
    input_tensor = preprocess(image)
    with torch.no_grad():
        output = model(input_tensor)
        # Handle different model output formats
        if isinstance(output, dict) and 'out' in output:
            output = output['out']
    output_predictions = torch.argmax(output.squeeze(), dim=0).cpu().numpy()
    return output_predictions  # shape: [H, W]

def decode_segmap(mask):
    h, w = mask.shape
    output = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in color_map.items():
        output[mask == class_id] = color
    return Image.fromarray(output)

def segment_video(model, video_bytes, alpha=0.5, resize_to=(640, 320)):
    temp_video_path = "temp_video.mp4"
    with open(temp_video_path, "wb") as f:
        f.write(video_bytes.read())

    cap = cv2.VideoCapture(temp_video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        mask = segment_image(model, img)
        decoded = decode_segmap(mask)
        overlay = Image.blend(img.resize(decoded.size), decoded, alpha=alpha)
        overlay = overlay.resize(resize_to)

        frame_np = np.array(overlay)
        frames.append(frame_np)

        yield int((i + 1) / total_frames * 100), frames
    cap.release()

def decode_segmap(mask, hidden_classes=None):
    if hidden_classes is None:
        hidden_classes = []
    h, w = mask.shape
    output = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in color_map.items():
        if class_id in hidden_classes:
            continue  
        output[mask == class_id] = color
    return Image.fromarray(output)



def build_model(model_name):
    """
    Recreates a segmentation model architecture without loading weights.
    Used for rebuilding models when loading state_dicts from Hugging Face.
    """
    if model_name == "hrnetv2_w48":
        args = argparse.Namespace(cfg="config/hrnet_config.yaml", opts=['DATASET.NUM_CLASSES', '19'])
        update_config(hrnet_config, args)
        model = HighResolutionNet(hrnet_config)
        model.last_layer = torch.nn.Conv2d(in_channels=720, out_channels=19, kernel_size=1)
        return model

    elif "resnet101" in model_name:
        model = models.segmentation.deeplabv3_resnet101(pretrained=False, aux_loss=None)
        model.classifier[4] = torch.nn.Sequential(
            torch.nn.Dropout(0.3),
            torch.nn.Conv2d(256, 19, kernel_size=1)
        )
        return model

    else:
        raise ValueError(f"Unknown architecture: {model_name}")

