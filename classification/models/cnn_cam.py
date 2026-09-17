import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class FrozenResNet18CamGenerator(nn.Module):


    def __init__(self, checkpoint, target_class=0):
        super().__init__()
        self.checkpoint = checkpoint
        self.target_class = int(target_class)
        self.cnn = models.resnet18(weights=None)
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, 2)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state_dict = payload.get("state_dict", payload)
        self.cnn.load_state_dict(state_dict, strict=True)
        for parameter in self.parameters():
            parameter.requires_grad = False
        self.eval()

    def train(self, mode=True):

        super().train(False)
        self.cnn.eval()
        return self

    @torch.no_grad()
    def forward(self, images, output_size):
        network = self.cnn
        x = network.conv1(images)
        x = network.bn1(x)
        x = network.relu(x)
        x = network.maxpool(x)
        x = network.layer1(x)
        x = network.layer2(x)
        x = network.layer3(x)
        features = network.layer4(x)
        class_weight = network.fc.weight[self.target_class]
        cam = torch.einsum("bchw,c->bhw", features, class_weight)
        cam = F.relu(cam).unsqueeze(1)
        cam = F.interpolate(
            cam, size=output_size, mode="bilinear", align_corners=False)
        cam = cam.flatten(1)
        cam_min = cam.amin(dim=1, keepdim=True)
        cam_max = cam.amax(dim=1, keepdim=True)
        return (cam - cam_min) / (cam_max - cam_min).clamp_min(1e-6)
