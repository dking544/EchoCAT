# EchoCAT

EchoCAT is an image preprocessing and classification project for fetal echocardiography, covering the complete pipeline from ultrasound image segmentation, region-of-interest cropping, grayscaling, and image enhancement, to view recognition, disease classification, fetal-level aggregation inference, and Grad-CAM visualization.

> This project is intended for research and experimental validation only and should not be used directly for clinical diagnosis or treatment decisions.

## Feature Overview

- **Image Preprocessing**
  - Generate ultrasound region segmentation masks based on TransUNet
  - Crop original images based on masks while preserving directory structure
  - Convert images to three-channel grayscale images
  - Optional image denoising/enhancement based on Noise2Same
- **Image Classification**
  - `view`: Four-chamber (4CH) vs. non-four-chamber view classification
  - `binary`: Normal vs. abnormal binary classification
  - `six`: Six-class diagnosis
- **Inference and Analysis**
  - Single-image inference
  - Evaluation on labeled image lists
  - Fetal-level prediction aggregation from multiple images
  - Grad-CAM visualization

## Project Structure

```text
EchoCAT/
├── preprocess/
│   ├── transunet/          # Segmentation, mask cropping, and grayscaling
│   └── enchancer/          # Noise2Same image enhancement (directory name retains existing spelling)
├── classification/
│   ├── recipes/            # Configurations for view, binary, and six tasks
│   ├── weights/            # Fixed classification weights and teacher model weights (Git LFS)
│   ├── train.py            # Classification training entry point
│   ├── predict.py          # Single-image inference and test set evaluation
│   ├── fetal.py            # Fetal-level aggregation inference
│   └── gradcam.py          # Grad-CAM visualization
└── README.md
```

## Environment Setup

It is recommended to create separate virtual environments for the classification module and the preprocessing module. `classification` uses newer PyTorch dependencies, while the Noise2Same dependencies retained in `preprocess/enchancer` are based on older versions; mixing them may cause version conflicts.

### 1. Obtain Code and Classification Weights

Classification weights are managed by Git LFS. Please install Git LFS first, then clone and pull large files:

```bash
git lfs install
git clone https://github.com/dking544/EchoCAT.git
cd EchoCAT
git lfs pull
```

If the repository has already been cloned, you can directly run `git lfs pull` in the repository root.

### 2. Classification Environment

Python 3.10 is recommended for the classification module:

```bash
cd classification
python3.10 -m venv .venv
source .venv/bin/activate

# NVIDIA CUDA 12.6
python -m pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
```

For CPU environments, replace `cu126` in the PyTorch installation source with `cpu`, and use `--device cpu` when running commands. On Windows PowerShell, activate the environment using `.venv\Scripts\Activate.ps1`.

### 3. Preprocessing Environment

TransUNet and Noise2Same can be installed separately to reduce dependency conflicts.

The main dependencies required by TransUNet include PyTorch, torchvision, timm, OpenCV, Pillow, NumPy, and tqdm:

```bash
cd preprocess/transunet
python -m pip install torch torchvision timm opencv-python Pillow numpy tqdm
```

The Noise2Same enhancement module uses the legacy dependency list provided in the repository and is recommended to be installed in a separate compatible environment:

```bash
cd preprocess/enchancer
python -m pip install -r requirements.txt
```

## Model Weights

The fixed classification weights and teacher model weights are located in `classification/weights/` and can be used after correctly running `git lfs pull`.

Preprocessing model weights are not yet provided in the repository:

| Module | Purpose | Download Link |
|---|---|---|
| TransUNet | Ultrasound region segmentation | https://mega.nz/folder/wZwkgKbC#kOn6ZNzxhKrtjnZ_ALNZsQ |
| Noise2Same UNet | Image denoising/enhancement | https://mega.nz/folder/wZwkgKbC#kOn6ZNzxhKrtjnZ_ALNZsQ |
| Classification Model | Ultrasound image classification | https://mega.nz/folder/BVBxGYJR#vtqSL5IZ7XoHa-xZAWrABQ |

After downloading, they can be placed in a custom directory and their paths specified via command-line arguments or `preprocess/enchancer/config/experiment/custom.yaml`. Below, `<TRANSUNET_WEIGHT>` and `<NOISE2SAME_WEIGHT>` are used to denote the corresponding weight files.

## Preprocessing Pipeline

The recommended pipeline is as follows:

```text
Raw images -> TransUNet segmentation -> Mask cropping -> Three-channel grayscaling -> Noise2Same enhancement (optional)
```

The following commands are all executed from the corresponding module directories.

### 1. Generate Segmentation Masks

```bash
cd preprocess/transunet

python transunetval.py \
  --model_path <TRANSUNET_WEIGHT> \
  --input_folder /path/to/raw_images \
  --output_folder /path/to/masks \
  --device cuda:0 \
  --mode progress
```

The default binarization threshold is `0.5`, which can be adjusted via `--threshold`. Example of single-image inference:

```bash
python transunetval.py \
  --model_path <TRANSUNET_WEIGHT> \
  --mode single \
  --single_image /path/to/image.png \
  --single_output /path/to/mask.png \
  --device cuda:0
```

### 2. Crop Original Images Based on Masks

```bash
python cutbyunet.py \
  --original_root_folder /path/to/raw_images \
  --segmented_root_folder /path/to/masks \
  --output_root_folder /path/to/cropped_images \
  --coverage 0.70 \
  --padding_pixels 0
```

The script recursively processes subdirectories, maps mask coordinates back to the original image dimensions, and preserves the relative directory structure in the output directory. `--coverage` indicates the foreground pixel coverage ratio used to determine the cropping region.

### 3. Convert to Three-Channel Grayscale Images

```bash
python grayscale.py /path/to/cropped_images /path/to/grayscale_images
```

If the output directory is omitted, the script will generate files with a `_gray` suffix alongside the original images.

### 4. Noise2Same Image Enhancement (Optional)

Edit `preprocess/enchancer/config/experiment/custom.yaml` and set at least:

```yaml
data:
  train_path: /path/to/grayscale_images

checkpoint_path: /path/to/noise2same_weight.pth
output_dir: /path/to/enhanced_images
```

Then run:

```bash
cd preprocess/enchancer
python test.py +backbone=unet +experiment=custom
```

The output will preserve the relative directory structure of the input data as much as possible. To train the enhancement model, run with the same configuration:

```bash
python train.py +backbone=unet +experiment=custom
```

## Classification Tasks and Labels

| Task | Label Definition |
|---|---|
| `view` | `0`: 4CH; `1`: Non-4CH |
| `binary` | `0`: Normal; `1`: Abnormal |
| `six` | `0`: Normal; `1`: Single Ventricle; `2`: Septal Defect; `3`: Ebstein Anomaly; `4`: Cardiac Tumor; `5`: Ventricular Disproportion |

Training, validation, and test lists are all UTF-8 text files, each line containing an image path and a label separated by a space:

```text
images/example_001.png 0
images/example_002.png 1
```

Relative image paths are based on the directory where the list file is located. To avoid data leakage, all images from the same fetus should be placed in the same data split.

## Classification Training

The following commands are executed from the `classification/` directory:

```bash
cd classification

# Four-chamber view recognition
python train.py \
  --task view \
  --train-list data/view_train.txt \
  --val-list data/view_val.txt \
  --output runs/view \
  --device cuda:0

# Normal/abnormal binary classification
python train.py \
  --task binary \
  --train-list data/binary_train.txt \
  --val-list data/binary_val.txt \
  --output runs/binary \
  --device cuda:0

# Six-class diagnosis
python train.py \
  --task six \
  --train-list data/six_train.txt \
  --val-list data/six_val.txt \
  --output runs/six \
  --device cuda:0
```

The default training runs for `100` epochs. The output directory will save the best validation weights, final weights, training logs, and `checkpoints.json`. To resume training, use the best checkpoint recorded therein:

```bash
python train.py \
  --task six \
  --train-list data/six_train.txt \
  --val-list data/six_val.txt \
  --output runs/six \
  --resume-best runs/six/epoch_N.pth \
  --device cuda:0
```

Binary and six-class classification also support optional early stopping monitoring; for complete parameters and fetal-level validation mapping format, see [`classification/README.md`](classification/README.md).

## Image Inference and Evaluation

When `--checkpoint` is not specified, the program will automatically use the fixed weights for the corresponding task in `classification/weights/`.

Single-image inference:

```bash
cd classification
python predict.py \
  --task binary \
  --image data/example.png \
  --output results/example.json \
  --device cuda:0
```

Evaluate a labeled test list:

```bash
python predict.py \
  --task six \
  --list data/six_test.txt \
  --protocol validation \
  --output results/six_test.json \
  --device cuda:0
```

When using self-trained weights, add `--checkpoint runs/six/epoch_N.pth`.

## Fetal-Level Inference

Fetal-level inference first filters 4CH images, then aggregates diagnosis probabilities for the same fetus. Create a JSONL manifest with one record per line:

```json
{"image":"images/example_001.png","fetus_id":"fetus_001","fetus_label":1}
{"image":"images/example_002.png","fetus_id":"fetus_001","fetus_label":1}
```

`fetus_label` is an optional fetal-level reference label. Run:

```bash
cd classification

python fetal.py \
  --task binary \
  --manifest data/fetal.jsonl \
  --output results/fetal_binary.json \
  --device cuda:0
```

Change `--task` to `six` to perform six-class fetal-level inference.

## Grad-CAM

```bash
cd classification

python gradcam.py \
  --task six \
  --image data/example.png \
  --output results/gradcam \
  --device cuda:0
```

By default, the predicted class is explained; a target class can be specified via `--target CLASS_INDEX`. The output directory contains preprocessed images, overlay images, heatmap arrays, and class scores.

## Testing

Run the release tests in the classification environment:

```bash
cd classification
python -m tests.test_release
```

## Notes

- Input images, annotations, and training data are not included in this repository.
- Preprocessing weight download links will be added in the future.
- For more training details, early stopping rules, and data formats for the classification module, see [`classification/README.md`](classification/README.md).