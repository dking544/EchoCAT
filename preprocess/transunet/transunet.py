import os

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import glob
import json

class MedicalImagePredictDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(image_file).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, image_file
class MedicalImageDataset(Dataset):
    def __init__(self, image_dir, json_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.data_info = self.load_annotations(json_file)
        print(f"Loaded {len(self.data_info)} pairs of images and masks.")

    def load_annotations(self, json_dir):
        data_info = []
        json_files = glob.glob(os.path.join(json_dir, '*.json'))  # 获取所有JSON文件路径

        for json_file in json_files:
            with open(json_file, 'r') as f:
                data = json.load(f)

            for image in data['images']:
                image_id = image['id']
                file_name = image['file_name'].split('/')[-1]  # 获取文件名
                mask = np.zeros((image['height'], image['width']), dtype=np.uint8)

                # 生成对应的mask
                for annotation in data['annotations']:
                    if annotation['image_id'] == image_id:
                        category_id = annotation['category_id']
                        segmentation = annotation['segmentation']
                        mask = self.rle2mask(segmentation, mask.shape, category_id)
                        mask = mask * 255

                data_info.append({
                    'image': os.path.join(self.image_dir, file_name),
                    'mask': mask
                })

        return data_info

    def rle2mask(self, rle, shape, category_id):
        # 将RLE编码的掩码转换为二值掩码
        # 这里需要根据你的具体RLE编码格式来实现转换逻辑
        # 以下是一个简化的示例，实际应用中需要根据RLE格式进行解析
        mask = np.zeros(shape, dtype=np.uint8)
        for segment in rle:
            # 假设segment是[x, y, x, y, ...]
            points = np.array(segment).reshape(-1, 2)
            polygon = np.array(points, np.int32)
            cv2.fillPoly(mask, [polygon], 1)
        return mask

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        image_info = self.data_info[idx]
        image = Image.open(image_info['image']).convert('RGB')
        mask = Image.fromarray(image_info['mask'])

        if self.transform:
            image = img_transform(image)
            mask = mask_transform(mask)

        return image, mask



# ---------------------------- CNN Encoder ----------------------------
class CNNEncoder(nn.Module):
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2, 2)

        self.encoder2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2, 2)

        self.encoder3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(2, 2)

        self.encoder4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.encoder1(x)
        x2 = self.encoder2(self.pool1(x1))
        x3 = self.encoder3(self.pool2(x2))
        x4 = self.encoder4(self.pool3(x3))
        return x1, x2, x3, x4

# ---------------------------- Vision Transformer ----------------------------
class TransEncoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=512, embed_dim=512, depth=6, num_heads=8):
        super(TransEncoder, self).__init__()
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, x):
        x = self.patch_embed(x)  # [B, C, H/patch, W/patch]
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, N, C]
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.transformer(x)
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return x

# ---------------------------- Decoder ----------------------------
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(256, 128)

        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(128, 64)

        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x4, x3, x2, x1):
        x = self.up1(x4)
        x = self.dec1(torch.cat([x, x3], dim=1))
        x = self.up2(x)
        x = self.dec2(torch.cat([x, x2], dim=1))
        x = self.up3(x)
        x = self.dec3(torch.cat([x, x1], dim=1))
        return self.final(x)

# ---------------------------- TransUNet ----------------------------
class TransUNet(nn.Module):
    def __init__(self, img_size=224, vit_patch_size=16, vit_embed_dim=512):
        super(TransUNet, self).__init__()
        self.encoder = CNNEncoder()
        self.vit = TransEncoder(
            img_size=img_size,
            patch_size=vit_patch_size,
            in_chans=512,
            embed_dim=vit_embed_dim
        )
        self.vit_reshape = nn.Sequential(
            nn.Conv2d(vit_embed_dim, 512, 1),
            nn.Upsample(scale_factor=vit_patch_size, mode='bilinear', align_corners=True)
        )
        self.decoder = Decoder()

    def forward(self, x):
        x1, x2, x3, x4 = self.encoder(x)
        vit_out = self.vit(x4)
        vit_out = self.vit_reshape(vit_out)
        return self.decoder(vit_out, x3, x2, x1)

# ---------------------------- Prediction Utility ----------------------------
def predict(model, dataloader, device, output_dir):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    with torch.no_grad():
        for idx, (images, _) in enumerate(dataloader):
            images = images.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs).cpu().numpy() > 0.5
            for i in range(images.size(0)):
                mask = (preds[i][0] * 255).astype(np.uint8)
                img = Image.fromarray(mask)
                img.save(os.path.join(output_dir, f"pred_{idx * dataloader.batch_size + i}.png"))

# ---------------------------- Usage Example ----------------------------
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = MedicalImagePredictDataset("/home/UserData/wyo/data/traingan/image/400", "/home/UserData/wyo/data/traingan/json400", transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransUNet(vit_embed_dim=512).to(device)

    # Load pretrained weights if any
    # model.load_state_dict(torch.load("transunet.pth"))

    predict(model, dataloader, device, "./outputs")

