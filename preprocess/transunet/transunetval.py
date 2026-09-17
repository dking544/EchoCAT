import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from torchvision import models
from timm.models.vision_transformer import VisionTransformer
from timm.models.vision_transformer import trunc_normal_
from tqdm import tqdm
import cv2


# ------------------------- 模型定义 (与训练时相同) -------------------------
class TransEncoder(VisionTransformer):
    def __init__(self, img_size=7, patch_size=1, in_chans=512, embed_dim=768,
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
            # representation_size=None,
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


# ------------------------- 推理类 -------------------------
class TransUNetInference:
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # 加载模型
        self.model = TransUNet(n_classes=1).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print(f"Model loaded from {model_path}")

        # 图像预处理 - 与训练时保持一致
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.2283886,  0.19033915, 0.15560639],
                                 [0.25484519, 0.21482847, 0.1884323])
        ])

    def predict_single_image(self, image_path, threshold=0.5):
        """
        预测单张图像
        Args:
            image_path: 图像路径
            threshold: 二值化阈值
        Returns:
            mask: 预测的掩码 (0-255)
        """
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        original_size = image.size  # (width, height)

        # 预处理
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # 推理
        with torch.no_grad():
            output = self.model(input_tensor)
            # 应用sigmoid激活
            prob = torch.sigmoid(output).squeeze(0).squeeze(0)  # [224, 224]

            # 二值化
            mask = (prob > threshold).float()

            # 转换为numpy
            mask_np = mask.cpu().numpy()

            # 恢复原始尺寸
            mask_resized = cv2.resize(mask_np, original_size, interpolation=cv2.INTER_NEAREST)

            # 转换为0-255
            mask_final = (mask_resized * 255).astype(np.uint8)

        return mask_final

    def process_folder(self, input_folder, output_folder, threshold=0.5,
                       supported_formats=None):
        """
        处理整个文件夹
        Args:
            input_folder: 输入文件夹路径
            output_folder: 输出文件夹路径
            threshold: 二值化阈值
            supported_formats: 支持的图像格式
        """
        if supported_formats is None:
            supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']

        # 创建输出文件夹
        os.makedirs(output_folder, exist_ok=True)

        # 遍历输入文件夹的所有子文件夹
        for root, dirs, files in os.walk(input_folder):
            for file in files:
                # 检查文件格式
                if any(file.lower().endswith(fmt) for fmt in supported_formats):
                    # 输入文件路径
                    input_path = os.path.join(root, file)

                    # 计算相对路径
                    rel_path = os.path.relpath(root, input_folder)

                    # 输出文件夹路径
                    output_dir = os.path.join(output_folder, rel_path)
                    os.makedirs(output_dir, exist_ok=True)

                    # 输出文件路径 (添加_mask后缀)
                    name_without_ext = os.path.splitext(file)[0]
                    output_path = os.path.join(output_dir, f"{name_without_ext}.jpg")

                    # 检查是否已存在
                    if os.path.exists(output_path):
                        print(f"跳过已存在的文件: {output_path}")
                        continue

                    try:
                        # 预测
                        mask = self.predict_single_image(input_path, threshold)

                        # 保存掩码
                        cv2.imwrite(output_path, mask)
                        print(f"处理完成: {input_path} -> {output_path}")

                    except Exception as e:
                        print(f"处理失败: {input_path}, 错误: {str(e)}")
                        continue

    def batch_process_with_progress(self, input_folder, output_folder, threshold=0.5):
        """
        带进度条的批量处理
        """
        supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']

        # 统计总文件数
        total_files = 0
        file_list = []

        for root, dirs, files in os.walk(input_folder):
            for file in files:
                if any(file.lower().endswith(fmt) for fmt in supported_formats):
                    input_path = os.path.join(root, file)
                    rel_path = os.path.relpath(root, input_folder)
                    file_list.append((input_path, rel_path, file))
                    total_files += 1

        print(f"找到 {total_files} 个图像文件")

        # 创建输出文件夹
        os.makedirs(output_folder, exist_ok=True)

        # 使用tqdm显示进度
        with tqdm(total=total_files, desc="处理图像") as pbar:
            for input_path, rel_path, file in file_list:
                # 输出文件夹路径
                output_dir = os.path.join(output_folder, rel_path)
                os.makedirs(output_dir, exist_ok=True)

                # 输出文件路径
                name_without_ext = os.path.splitext(file)[0]
                output_path = os.path.join(output_dir, f"{name_without_ext}.jpg")

                # 检查是否已存在
                if os.path.exists(output_path):
                    pbar.set_postfix({"状态": "跳过", "文件": file})
                    pbar.update(1)
                    continue

                try:
                    # 预测
                    mask = self.predict_single_image(input_path, threshold)

                    # 保存掩码
                    cv2.imwrite(output_path, mask)
                    pbar.set_postfix({"状态": "完成", "文件": file})

                except Exception as e:
                    pbar.set_postfix({"状态": "失败", "文件": file, "错误": str(e)[:20]})

                pbar.update(1)


# ------------------------- 使用示例 -------------------------
if __name__ == "__main__":
    # 示例1: 直接使用
    model_path = ""
    input_folder = ""
    output_folder = ""

    # 创建推理器
    inferencer = TransUNetInference(model_path, device='cuda')

    # # 方法1: 简单批量处理
    # print("开始批量处理...")
    # inferencer.process_folder(input_folder, output_folder, threshold=0.5)

    # 方法2: 带进度条的批量处理
    print("开始带进度条的批量处理...")
    inferencer.batch_process_with_progress(input_folder, output_folder, threshold=0.5)

    # # 方法3: 单张图像处理
    # single_image_path = "/path/to/single/image.jpg"
    # if os.path.exists(single_image_path):
    #     mask = inferencer.predict_single_image(single_image_path, threshold=0.5)
    #     cv2.imwrite("single_result_mask.png", mask)
    #     print("单张图像处理完成")

    print("所有处理完成！")


# ------------------------- 额外功能 -------------------------
def visualize_results(image_path, mask_path, output_path):
    """可视化原图和掩码的对比"""
    # 读取原图和掩码
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # 创建彩色掩码
    mask_colored = cv2.applyColorMap(mask, cv2.COLORMAP_JET)

    # 叠加显示
    overlay = cv2.addWeighted(image, 0.7, mask_colored, 0.3, 0)

    # 拼接显示
    result = np.hstack([image, mask_colored, overlay])

    # 保存结果
    cv2.imwrite(output_path, result)
    print(f"可视化结果保存到: {output_path}")


def batch_visualize(input_folder, mask_folder, output_folder):
    """批量可视化"""
    os.makedirs(output_folder, exist_ok=True)

    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                # 原图路径
                image_path = os.path.join(root, file)

                # 掩码路径
                rel_path = os.path.relpath(root, input_folder)
                mask_name = os.path.splitext(file)[0] + ".jpg"
                mask_path = os.path.join(mask_folder, rel_path, mask_name)

                # 输出路径
                output_dir = os.path.join(output_folder, rel_path)
                os.makedirs(output_dir, exist_ok=True)
                vis_name = os.path.splitext(file)[0] + "_visualization.png"
                output_path = os.path.join(output_dir, vis_name)

                if os.path.exists(mask_path):
                    visualize_results(image_path, mask_path, output_path)
                else:
                    print(f"掩码文件不存在: {mask_path}")

# 使用示例:
# python transunet_inference.py