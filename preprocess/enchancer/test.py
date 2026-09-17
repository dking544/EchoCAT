import os
from pathlib import Path

import hydra
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from PIL import Image

from noise2same import model, util
from noise2same.dataset.getter import get_test_dataset
from utils import parametrize_backbone_and_head


def save_image(tensor, path):
    """将tensor保存为图片"""
    # tensor shape: (C, H, W) or (H, W)
    if isinstance(tensor, torch.Tensor):
        img = tensor.cpu().float().numpy()  # 转为float32
    else:
        img = np.array(tensor, dtype=np.float32)
    
    # 处理通道顺序: (C, H, W) -> (H, W, C)
    if img.ndim == 3 and img.shape[0] in [1, 3]:
        img = np.transpose(img, (1, 2, 0))
    
    # 移除单通道维度: (H, W, 1) -> (H, W)
    if img.ndim == 3 and img.shape[2] == 1:
        img = img.squeeze(-1)
    
    # 归一化到 0-255
    # 假设输入范围在 [-1, 1] 或 [0, 1]
    img_min, img_max = img.min(), img.max()
    if img_min < 0:  # 如果有负值，假设范围是 [-1, 1]
        img = (img + 1) / 2.0  # 转到 [0, 1]
    # 否则假设已经在 [0, 1]
    
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    
    # 保存
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(path)


def get_relative_path(dataset, idx, dataset_root=None):
    """从数据集中获取文件的相对路径"""
    try:
        file_path = None
        
        # 方法1: AbstractNoiseDataset - 使用images属性（噪声数据集的标准属性）
        if hasattr(dataset, 'images') and isinstance(dataset.images, list):
            file_path = dataset.images[idx]
            # images可能是字符串路径或numpy数组
            if isinstance(file_path, (str, Path)):
                file_path = Path(file_path)
            else:
                # 如果是numpy数组，说明图像已预加载，无法获取路径
                return None
        
        # 方法2: 直接从dataset获取文件路径列表
        elif hasattr(dataset, 'image_files'):
            file_path = dataset.image_files[idx]
        elif hasattr(dataset, 'files'):
            file_path = dataset.files[idx]
        elif hasattr(dataset, 'file_list'):
            file_path = dataset.file_list[idx]
        elif hasattr(dataset, 'data_paths'):
            file_path = dataset.data_paths[idx]
        elif hasattr(dataset, 'image_paths'):
            file_path = dataset.image_paths[idx]
        
        # 方法3: 通过dataset的samples属性
        elif hasattr(dataset, 'samples'):
            sample = dataset.samples[idx]
            file_path = sample[0] if isinstance(sample, tuple) else sample
        
        # 方法4: 如果dataset有imgs属性（ImageFolder格式）
        elif hasattr(dataset, 'imgs'):
            file_path = dataset.imgs[idx][0]
        
        # 方法5: 尝试调用dataset的内部方法
        elif hasattr(dataset, 'get_path'):
            file_path = dataset.get_path(idx)
        elif hasattr(dataset, '_get_path'):
            file_path = dataset._get_path(idx)
        elif hasattr(dataset, 'get_image_path'):
            file_path = dataset.get_image_path(idx)
        
        # 如果成功获取file_path，计算相对路径
        if file_path is not None:
            file_path = Path(file_path)
            
            # 尝试计算相对路径
            if dataset_root is not None:
                try:
                    return file_path.relative_to(dataset_root)
                except ValueError:
                    # file_path不在dataset_root下，可能使用了绝对路径
                    # 尝试使用path属性作为根目录
                    if hasattr(dataset, 'path'):
                        try:
                            return file_path.relative_to(Path(dataset.path))
                        except ValueError:
                            pass
            
            # 如果没有root或计算失败，返回文件名
            return Path(file_path.name)
        
    except Exception as e:
        print(f"Warning: Could not get relative path for index {idx}: {e}")
    
    return None


def get_dataset_root(dataset):
    """获取数据集的根目录"""
    return Path(dataset.path)


@hydra.main(config_path="config", config_name="config", version_base="1.1")
def main(cfg: DictConfig) -> None:
    # Backbone / experiment check
    if "backbone_name" not in cfg.keys():
        print("Please specify a backbone with `+backbone=name`")
        return
    if "experiment" not in cfg.keys():
        print("Please specify an experiment with `+experiment=name`")
        return

    print("==== Config ====")
    print(OmegaConf.to_yaml(cfg))
    cwd = Path(get_original_cwd())
    print(f"Working directory: {cwd}")

    util.fix_seed(cfg.seed)

    # -------------------------------
    # 加载测试数据集
    # -------------------------------
    print("Loading test dataset...")
    test_dataset, _ = get_test_dataset(cfg, cwd)
    
    # 获取数据集根目录
    dataset_root = get_dataset_root(test_dataset)
    if dataset_root:
        print(f"Dataset root directory: {dataset_root}")
    else:
        print("Warning: Could not determine dataset root directory")
    
    # 调试：打印数据集的属性
    print("\n==== Dataset Attributes ====")
    dataset_attrs = [attr for attr in dir(test_dataset) if not attr.startswith('_')]
    print(f"Available attributes: {dataset_attrs[:20]}")  # 打印前20个属性
    
    # 尝试获取第一个样本的路径信息
    if len(test_dataset) > 0:
        print("\n==== First Sample Path Info ====")
        rel_path = get_relative_path(test_dataset, 0, dataset_root)
        if rel_path:
            print(f"Sample 0 relative path: {rel_path}")
        else:
            print("Could not get path for sample 0")
            # 尝试打印更多调试信息
            if hasattr(test_dataset, '__dict__'):
                print(f"Dataset instance attributes: {list(test_dataset.__dict__.keys())}")
    print("=" * 40 + "\n")
    
    loader_test = DataLoader(
        test_dataset,
        batch_size=1,  # 测试时batch_size=1
        num_workers=cfg.training.num_workers,
        shuffle=False,
        pin_memory=True,
    )

    # -------------------------------
    # PSF
    # -------------------------------
    psf = getattr(test_dataset, "psf", None)
    if psf is None and getattr(cfg, "psf", None) is not None:
        psf = cwd / cfg.psf.path
        print(f"Read PSF from {psf}")

    # -------------------------------
    # 加载模型
    # -------------------------------
    print("Loading model...")
    backbone, head = parametrize_backbone_and_head(cfg)
    mdl = model.Noise2Same(
        n_dim=cfg.data.n_dim,
        in_channels=cfg.data.n_channels,
        psf=psf,
        psf_size=cfg.psf.psf_size if "psf" in cfg else None,
        psf_pad_mode=cfg.psf.psf_pad_mode if "psf" in cfg else None,
        psf_fft=cfg.psf.psf_fft if "psf" in cfg else None,
        backbone=backbone,
        head=head,
        **cfg.model,
    )

    # 加载训练好的权重
    checkpoint_path = cfg.get("checkpoint_path", None)
    if checkpoint_path is None:
        print("Warning: No checkpoint_path specified in config. Using untrained model.")
    else:
        checkpoint_path = cwd / checkpoint_path
        if checkpoint_path.exists():
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # 提取模型权重
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # 移除 'module.' 前缀（如果有DataParallel）
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            
            # 跳过 mask_kernel.kernel 这个有问题的参数
            if 'mask_kernel.kernel' in new_state_dict:
                print("Skipping mask_kernel.kernel due to memory sharing issue")
                del new_state_dict['mask_kernel.kernel']
            
            mdl.load_state_dict(new_state_dict, strict=False)
            print("Note: Loaded with strict=False to handle problematic parameters")
            print("Checkpoint loaded successfully")
        else:
            print(f"Warning: Checkpoint not found at {checkpoint_path}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mdl = mdl.to(device)
    mdl.eval()

    # -------------------------------
    # 创建输出根目录
    # -------------------------------
    output_root = Path(cfg.get("output_dir", "denoised_results"))
    output_root.mkdir(parents=True, exist_ok=True)
    print(f"Saving results to {output_root}")

    # -------------------------------
    # 推理并保存（保持原始文件结构）
    # -------------------------------
    print("Starting denoising...")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader_test, desc="Denoising")):
            # 获取输入图像
            if isinstance(batch, dict):
                if 'image' in batch:
                    noisy_img = batch['image']
                elif 'noisy' in batch:
                    noisy_img = batch['noisy']
                else:
                    noisy_img = batch[list(batch.keys())[0]]
            else:
                noisy_img = batch

            noisy_img = noisy_img.to(device)

            # 如果使用AMP
            if cfg.training.get('amp', False):
                with torch.cuda.amp.autocast():
                    output = mdl(noisy_img)
            else:
                output = mdl(noisy_img)
            
            # 处理模型输出
            if isinstance(output, tuple):
                if isinstance(output[1], dict) and 'image' in output[1]:
                    denoised_img = output[1]['image']
                else:
                    denoised_img = output[0] if output[0] is not None else output[1]
            else:
                denoised_img = output

            # 获取原始文件的相对路径
            rel_path = get_relative_path(test_dataset, i, dataset_root)
            
            if rel_path is not None:
                # 使用原始文件结构
                output_path = output_root / rel_path
                save_image(denoised_img[0], output_path)
                
                # 可选：保存原始噪声图像（在相同结构下）
                if cfg.get("save_noisy", False):
                    noisy_output_root = Path(cfg.get("noisy_output_dir", "noisy_results"))
                    noisy_output_path = noisy_output_root / rel_path
                    save_image(noisy_img[0], noisy_output_path)
            else:
                # 如果无法获取相对路径，使用fallback命名方式
                if i == 0:  # 只在第一次打印警告
                    print(f"\nWarning: Could not get relative path. Using fallback naming.")
                    print(f"Please check your dataset class and ensure it has one of these attributes:")
                    print(f"  - image_files, files, file_list, data_paths, image_paths")
                    print(f"  - samples, imgs")
                    print(f"  - get_path(), get_image_path() methods")
                    print(f"Or implement a get_path(idx) method in your dataset class.\n")
                
                output_path = output_root / f"denoised_{i:04d}.png"
                save_image(denoised_img[0], output_path)
                
                if cfg.get("save_noisy", False):
                    noisy_output_path = output_root / f"noisy_{i:04d}.png"
                    save_image(noisy_img[0], noisy_output_path)

    print(f"Denoising completed! Results saved to {output_root}")
    print(f"Output structure matches dataset structure at: {dataset_root}")


if __name__ == "__main__":
    main()