import torch
import torch.nn as nn
import numpy as np
import cv2

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 24, kernel_size=5, stride=2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(24 * 31 * 48, 1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

class MLController:
    def __init__(self, model_path, device):
        self.device = device
        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        model = CNNModel()
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        return model.to(self.device)

    def preprocess(self, image):
        image = cv2.resize(image, (200, 66))  # Resize for model input
        image = image / 255.0
        image = np.transpose(image, (2, 0, 1))  # CxHxW
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(self.device)
        return image

    def get_steering_angle(self, seg_image):
        input_tensor = self.preprocess(seg_image)
        with torch.no_grad():
            model_angle = self.model(input_tensor).item()

        # Lane center detection from segmentation (IDs: 6=RoadLine, 7=Road)
        lane_mask = cv2.inRange(seg_image, (6, 6, 6), (7, 7, 7))
        moments = cv2.moments(lane_mask)
        if moments['m00'] != 0:
            cx = int(moments['m10'] / moments['m00'])
            error = (cx - lane_mask.shape[1] // 2) / (lane_mask.shape[1] // 2)
            center_angle = -error
        else:
            center_angle = 0.0

        # Blend model and lane-following prediction
        blended_angle = 0.6 * model_angle + 0.4 * center_angle
        return np.clip(blended_angle, -0.5, 0.5)
