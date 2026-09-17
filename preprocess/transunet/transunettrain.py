import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from torchvision import models
from timm.models.vision_transformer import VisionTransformer
from timm.models.vision_transformer import trunc_normal_

# ------------------------- 数据集 -------------------------
class MedicalImageDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        
        # 递归获取所有图片文件路径，并按字母顺序排序，确保一一对应
        # 过滤掉非图片文件（如 .DS_Store 等）
        valid_ext = ('.png', '.jpg', '.jpeg', '.tif', '.bmp')
        self.image_paths = []
        for root, dirs, files in os.walk(image_dir):
            for file in files:
                if file.lower().endswith(valid_ext):
                    self.image_paths.append(os.path.relpath(os.path.join(root, file), image_dir))
        
        self.image_paths.sort()
        
        self.img_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.2283886,  0.19033915, 0.15560639],
                                 [0.25484519, 0.21482847, 0.1884323])
        ])
        
        # 掩码使用最近邻插值，防止边缘产生非 0/1 的中间值
        self.mask_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_rel_path = self.image_paths[idx]
        
        img_path = os.path.join(self.image_dir, img_rel_path)
        # 假设掩码文件夹中的文件名与原图完全一致，保持相同的相对路径结构
        mask_path = os.path.join(self.mask_dir, img_rel_path)

        # 读取原图和掩码
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L') # 掩码转为单通道灰度图

        return self.img_transform(image), self.mask_transform(mask)


# ------------------------- Transformer编码器 -------------------------
class TransEncoder(VisionTransformer):
    def __init__(self, img_size=14, patch_size=1, in_chans=512, embed_dim=768,
                 depth=8, num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            num_classes=0,
        )
        self.patch_embed = nn.Conv2d(in_chans, embed_dim,
                                     kernel_size=patch_size,
                                     stride=patch_size)
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, embed_dim))
        trunc_normal_(self.pos_embed, std=.02)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        trunc_normal_(self.cls_token, std=.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)  
        H_p, W_p = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)  

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed[:, :(1 + H_p * W_p), :]
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x


# ------------------------- 修复的TransUNet -------------------------
class TransUNet(nn.Module):
    def __init__(self, n_classes=1):
        super().__init__()
        self.n_classes = n_classes
        resnet = models.resnet34(pretrained=True)

        self.encoder1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  
        self.encoder2 = nn.Sequential(resnet.maxpool, resnet.layer1)  
        self.encoder3 = resnet.layer2  
        self.encoder4 = resnet.layer3  
        self.encoder5 = resnet.layer4  

        self.vit = TransEncoder(
            img_size=7,  
            patch_size=1,
            in_chans=512,
            embed_dim=768,
            depth=8,
            num_heads=12
        )

        self.upconv5 = nn.ConvTranspose2d(768, 256, kernel_size=2, stride=2)  
        self.decoder4 = nn.Sequential(
            nn.Conv2d(256 + 256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.upconv4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)  
        self.decoder3 = nn.Sequential(
            nn.Conv2d(128 + 128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)  
        self.decoder2 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.upconv2 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)  
        self.decoder1 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.final_upconv = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)  
        self.final_conv = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.encoder1(x)  
        x2 = self.encoder2(x1)  
        x3 = self.encoder3(x2)  
        x4 = self.encoder4(x3)  
        x5 = self.encoder5(x4)  

        vit_out = self.vit(x5)  
        vit_out = vit_out[:, 1:, :]  

        B, N, E = vit_out.shape
        H_p = W_p = int(N ** 0.5)  
        vit_out = vit_out.permute(0, 2, 1).contiguous().view(B, E, H_p, W_p)  

        d5 = self.upconv5(vit_out)  

        d4 = torch.cat([d5, x4], dim=1)
        d4 = self.decoder4(d4)

        d4_up = self.upconv4(d4)  

        d3 = torch.cat([d4_up, x3], dim=1)
        d3 = self.decoder3(d3)

        d3_up = self.upconv3(d3)  

        d2 = torch.cat([d3_up, x2], dim=1)
        d2 = self.decoder2(d2)

        d2_up = self.upconv2(d2)  

        d1 = torch.cat([d2_up, x1], dim=1)
        d1 = self.decoder1(d1)

        d1_up = self.final_upconv(d1)  
        out = self.final_conv(d1_up)  

        return out


# ------------------------- 训练函数 -------------------------
def train(model, dataloader, num_epochs, device, ckpt_path=None):
    if ckpt_path is not None and os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)
        print(f"Loaded checkpoint from {ckpt_path}")
        
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_idx, (images, masks) in enumerate(dataloader):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks.float())
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if batch_idx == 0:  
                print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "west_china_segment.pth")
    print("Model saved.")


# ------------------------- 主函数 -------------------------
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 修改为直接指向存放成对图片的文件夹
    image_dir = ''  # 你的原始图像路径
    mask_dir = ''    # 你的掩码图像路径

    dataset = MedicalImageDataset(image_dir, mask_dir)
    print(f"Total paired images found: {len(dataset)}")
    
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

    model = TransUNet().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    train(model, dataloader, num_epochs=200, device=device)