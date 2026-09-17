# EchoCAT

EchoCAT 是一个面向胎儿超声心动图的图像预处理与分类项目，覆盖从超声图像分割、感兴趣区域裁剪、灰度化和图像增强，到切面识别、疾病分类、胎儿级聚合推理及 Grad-CAM 可视化的完整流程。

> 本项目仅用于科研与实验验证，不应直接用于临床诊断或治疗决策。

## 功能概览

- **图像预处理**
  - 基于 TransUNet 生成超声区域分割掩码
  - 根据掩码裁剪原始图像并保留目录结构
  - 将图像转换为三通道灰度图
  - 基于 Noise2Same 进行可选的图像去噪/增强
- **图像分类**
  - `view`：四腔心切面（4CH）与非四腔心切面分类
  - `binary`：正常与异常二分类
  - `six`：六分类诊断
- **推理与分析**
  - 单张图像推理
  - 带标签图像列表评估
  - 多张图像的胎儿级预测聚合
  - Grad-CAM 可视化

## 项目结构

```text
EchoCAT/
├── preprocess/
│   ├── transunet/          # 分割、掩码裁剪与灰度化
│   └── enchancer/          # Noise2Same 图像增强（目录名沿用现有拼写）
├── classification/
│   ├── recipes/            # view、binary、six 三项任务配置
│   ├── weights/            # 分类固定权重与教师模型权重（Git LFS）
│   ├── train.py            # 分类训练入口
│   ├── predict.py          # 单图推理与测试集评估
│   ├── fetal.py            # 胎儿级聚合推理
│   └── gradcam.py          # Grad-CAM 可视化
└── README.md
```

## 环境配置

建议为分类模块和预处理模块分别创建虚拟环境。`classification` 使用较新的 PyTorch 依赖，而 `preprocess/enchancer` 中保留的 Noise2Same 依赖基于较旧版本，混装可能产生版本冲突。

### 1. 获取代码与分类权重

分类权重由 Git LFS 管理。请先安装 Git LFS，再克隆并拉取大文件：

```bash
git lfs install
git clone https://github.com/dking544/EchoCAT.git
cd EchoCAT
git lfs pull
```

若已经克隆仓库，可直接在仓库根目录执行 `git lfs pull`。

### 2. 分类环境

分类模块建议使用 Python 3.10：

```bash
cd classification
python3.10 -m venv .venv
source .venv/bin/activate

# NVIDIA CUDA 12.6
python -m pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
```

CPU 环境可将 PyTorch 安装源中的 `cu126` 替换为 `cpu`，并在运行命令时使用 `--device cpu`。Windows PowerShell 使用 `.venv\Scripts\Activate.ps1` 激活环境。

### 3. 预处理环境

TransUNet 与 Noise2Same 可以分开安装，以减少依赖冲突。

TransUNet 所需的主要依赖包括 PyTorch、torchvision、timm、OpenCV、Pillow、NumPy 和 tqdm：

```bash
cd preprocess/transunet
python -m pip install torch torchvision timm opencv-python Pillow numpy tqdm
```

Noise2Same 增强模块使用仓库中提供的旧版依赖清单，建议在独立的兼容环境中安装：

```bash
cd preprocess/enchancer
python -m pip install -r requirements.txt
```

## 模型权重

分类固定权重和教师模型权重位于 `classification/weights/`，正确执行 `git lfs pull` 后即可使用。

预处理模型权重尚未在仓库中提供：

| 模块 | 用途 | 下载链接 |
|---|---|---|
| TransUNet | 超声区域分割 | https://mega.nz/folder/wZwkgKbC#kOn6ZNzxhKrtjnZ_ALNZsQ |
| Noise2Same UNet | 图像去噪/增强 | https://mega.nz/folder/wZwkgKbC#kOn6ZNzxhKrtjnZ_ALNZsQ |

下载后可放在自定义目录中，并通过命令行参数或 `preprocess/enchancer/config/experiment/custom.yaml` 指定路径。下文分别使用 `<TRANSUNET_WEIGHT>` 和 `<NOISE2SAME_WEIGHT>` 表示对应权重文件。

## 预处理流程

推荐流程如下：

```text
原始图像 -> TransUNet 分割 -> 掩码裁剪 -> 三通道灰度化 -> Noise2Same 增强（可选）
```

以下命令均从对应模块目录执行。

### 1. 生成分割掩码

```bash
cd preprocess/transunet

python transunetval.py \
  --model_path <TRANSUNET_WEIGHT> \
  --input_folder /path/to/raw_images \
  --output_folder /path/to/masks \
  --device cuda:0 \
  --mode progress
```

默认二值化阈值为 `0.5`，可通过 `--threshold` 调整。单图推理示例：

```bash
python transunetval.py \
  --model_path <TRANSUNET_WEIGHT> \
  --mode single \
  --single_image /path/to/image.png \
  --single_output /path/to/mask.png \
  --device cuda:0
```

### 2. 根据掩码裁剪原图

```bash
python cutbyunet.py \
  --original_root_folder /path/to/raw_images \
  --segmented_root_folder /path/to/masks \
  --output_root_folder /path/to/cropped_images \
  --coverage 0.70 \
  --padding_pixels 0
```

脚本会递归处理子目录，将掩码坐标映射回原图尺寸，并在输出目录中保留相对目录结构。`--coverage` 表示用于确定裁剪区域的前景像素覆盖比例。

### 3. 转换为三通道灰度图

```bash
python grayscale.py /path/to/cropped_images /path/to/grayscale_images
```

若省略输出目录，脚本会在原图旁生成带 `_gray` 后缀的文件。

### 4. Noise2Same 图像增强（可选）

编辑 `preprocess/enchancer/config/experiment/custom.yaml`，至少设置：

```yaml
data:
  train_path: /path/to/grayscale_images

checkpoint_path: /path/to/noise2same_weight.pth
output_dir: /path/to/enhanced_images
```

然后执行：

```bash
cd preprocess/enchancer
python test.py +backbone=unet +experiment=custom
```

输出会尽量保持输入数据的相对目录结构。需要训练增强模型时，可使用相同配置运行：

```bash
python train.py +backbone=unet +experiment=custom
```

## 分类任务与标签

| 任务 | 标签定义 |
|---|---|
| `view` | `0`: 4CH；`1`: Non-4CH |
| `binary` | `0`: Normal；`1`: Abnormal |
| `six` | `0`: Normal；`1`: Single Ventricle；`2`: Septal Defect；`3`: Ebstein Anomaly；`4`: Cardiac Tumor；`5`: Ventricular Disproportion |

训练、验证和测试列表均为 UTF-8 文本文件，每行包含图像路径和标签，以空格分隔：

```text
images/example_001.png 0
images/example_002.png 1
```

相对图像路径以列表文件所在目录为基准。为避免数据泄漏，同一胎儿的图像应全部放入同一个数据划分。

## 分类训练

以下命令从 `classification/` 目录执行：

```bash
cd classification

# 四腔心切面识别
python train.py \
  --task view \
  --train-list data/view_train.txt \
  --val-list data/view_val.txt \
  --output runs/view \
  --device cuda:0

# 正常/异常二分类
python train.py \
  --task binary \
  --train-list data/binary_train.txt \
  --val-list data/binary_val.txt \
  --output runs/binary \
  --device cuda:0

# 六分类诊断
python train.py \
  --task six \
  --train-list data/six_train.txt \
  --val-list data/six_val.txt \
  --output runs/six \
  --device cuda:0
```

默认训练 `100` 个 epoch。输出目录会保存最佳验证权重、最终权重、训练记录和 `checkpoints.json`。恢复训练时，使用其中记录的最佳 checkpoint：

```bash
python train.py \
  --task six \
  --train-list data/six_train.txt \
  --val-list data/six_val.txt \
  --output runs/six \
  --resume-best runs/six/epoch_N.pth \
  --device cuda:0
```

二分类与六分类还支持可选的早停监控；完整参数和胎儿级验证映射格式见 [`classification/README.md`](classification/README.md)。

## 图像推理与评估

不指定 `--checkpoint` 时，程序会自动使用 `classification/weights/` 中对应任务的固定权重。

单图推理：

```bash
cd classification
python predict.py \
  --task binary \
  --image data/example.png \
  --output results/example.json \
  --device cuda:0
```

评估带标签的测试列表：

```bash
python predict.py \
  --task six \
  --list data/six_test.txt \
  --protocol validation \
  --output results/six_test.json \
  --device cuda:0
```

使用自训练权重时添加 `--checkpoint runs/six/epoch_N.pth`。

## 胎儿级推理

胎儿级推理先筛选 4CH 图像，再聚合同一胎儿的诊断概率。创建 JSONL 清单，每行一条记录：

```json
{"image":"images/example_001.png","fetus_id":"fetus_001","fetus_label":1}
{"image":"images/example_002.png","fetus_id":"fetus_001","fetus_label":1}
```

`fetus_label` 为可选的胎儿级参考标签。运行：

```bash
cd classification

python fetal.py \
  --task binary \
  --manifest data/fetal.jsonl \
  --output results/fetal_binary.json \
  --device cuda:0
```

将 `--task` 改为 `six` 可执行六分类胎儿级推理。

## Grad-CAM

```bash
cd classification

python gradcam.py \
  --task six \
  --image data/example.png \
  --output results/gradcam \
  --device cuda:0
```

默认解释预测类别；可通过 `--target CLASS_INDEX` 指定目标类别。输出目录包含预处理图像、叠加图、热力图数组和类别分数。

## 测试

在分类环境中运行发布测试：

```bash
cd classification
python -m tests.test_release
```

## 说明

- 输入图像、标注和训练数据未包含在本仓库中。
- 预处理权重下载链接将在后续补充。
- 分类模块的更多训练细节、早停规则和数据格式见 [`classification/README.md`](classification/README.md)。
