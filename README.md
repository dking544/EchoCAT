# EchoCAT

Fetal ultrasound view classification and diagnostic classification, with training, image prediction, fetal-level inference, and Grad-CAM visualization.

## Installation

Use Python 3.10 and run commands from the repository root.

```bash
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows PowerShell (use instead of the command above)
.venv\Scripts\Activate.ps1

python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
```

For CPU-only installation, replace `cu126` with `cpu`. To run on CPU, use `--device cpu` instead of `--device cuda:0`.

## Classification Model Weights

The classification checkpoints are distributed separately from the GitHub code repository because of their file sizes.

**Download: [EchoCAT classification model weights](https://mega.nz/folder/BVBxGYJR#vtqSL5IZ7XoHa-xZAWrABQ)**

Download the files manually and extract them, if archived. Place the actual `.pth` files in the repository's `weights/` directory, keeping the filenames below unchanged. The scripts load local checkpoint files; they do not automatically download weights from MEGA. Do not pass the MEGA sharing-page URL as a checkpoint path.

```text
EchoCAT/
├── weights/
│   ├── view_fixed.pth
│   ├── binary_fixed.pth
│   ├── binary_teacher.pth
│   ├── six_fixed.pth
│   └── six_teacher.pth
├── examples/
│   ├── normal/
│   ├── single_ventricle/
│   ├── view.txt
│   ├── fetal_binary.jsonl
│   ├── fetal_six.jsonl
│   └── expected_results.json
└── run_examples.py
```

`view_fixed.pth`, `binary_fixed.pth`, and `six_fixed.pth` are the fixed checkpoints for view classification, binary diagnosis, and six-class diagnosis, respectively. The current implementation also reads `binary_teacher.pth` and `six_teacher.pth` when constructing the diagnostic models. Keep both teacher files in `weights/`, even though CAM guidance is not performed during inference. The teacher files are not substitutes for `view_fixed.pth`.


## Two Complete Fetal Examples

The `examples/` directory contains two deidentified complete fetal cases from WCSUH-2, rather than two individual images. Each case includes all of its preprocessed images in the evaluation cohort. These selected cases are provided for functional demonstration, not as a representative performance evaluation.

| Image folder | Images | Expected fetal binary diagnosis | Expected fetal six-class diagnosis |
|---|---:|---|---|
| `examples/normal/` | 35 | Normal | Normal |
| `examples/single_ventricle/` | 93 | Abnormal | Single Ventricle |

After installing the dependencies and placing all five checkpoints in `weights/`, run the following from the repository root:

```bash
python run_examples.py --device cpu
```

For an NVIDIA GPU:

```bash
python run_examples.py --device cuda:0
```

The command checks the example image inventory and checksums, runs image-level view classification on all 128 images, and then runs binary and six-class fetal-level inference. Results are written to `results/examples/`, with a summary in `results/examples/summary.json`. No model is trained or modified by this command.

Reference information is provided in `examples/expected_results.json`; use the 1030 version with the updated six-class checkpoint. The current runner compares the view predictions and final fetal diagnostic labels with the reference values. It records diagnostic probabilities but does not compare them element by element with the reference probabilities. Small numerical differences may occur across hardware and library versions.

The supplied images are already preprocessed classification inputs. This demonstration covers classification and fetal-level aggregation; it does not perform the preceding offline image preparation steps.

To run the three tasks separately:

```bash
python predict.py --task view --list examples/view.txt --protocol external --output results/examples/view.json --device cpu
python fetal.py --task binary --manifest examples/fetal_binary.jsonl --output results/examples/binary.json --device cpu
python fetal.py --task six --manifest examples/fetal_six.jsonl --output results/examples/six.json --device cpu
```

The fetal manifests contain both 4CH and non-4CH images so that the view model performs screening. Fetal reference diagnoses must not be treated as image-level diagnostic labels for non-4CH inputs.

## Tasks and Data Format

| Task | Class labels |
|---|---|
| `view` | 0: 4CH; 1: Non-4CH |
| `binary` | 0: Normal; 1: Abnormal |
| `six` | 0: Normal; 1: Single Ventricle; 2: Septal Defect; 3: Ebstein Anomaly; 4: Cardiac Tumor; 5: Ventricular Disproportion |

Training, validation, and test lists are UTF-8 text files with one image path and label per line:

```text
images/example_001.png 0
images/example_002.png 1
```

Relative image paths are resolved from the list file's directory. Keep images from the same fetus in one split. Use ultrasound images prepared consistently with the original experiments; the classification input transforms handle resizing, center cropping, and normalization.

## Training

```bash
# View classification
python train.py --task view --train-list data/view_train.txt --val-list data/view_val.txt --output runs/view --device cuda:0

# Binary diagnosis
python train.py --task binary --train-list data/binary_train.txt --val-list data/binary_val.txt --output runs/binary --device cuda:0

# Six-class diagnosis
python train.py --task six --train-list data/six_train.txt --val-list data/six_val.txt --output runs/six --device cuda:0
```

Training runs for 100 epochs by default and retains the best validation checkpoint and the final checkpoint. See `checkpoints.json` in the output directory for the saved epochs.

To resume from the best checkpoint, append the following to the original training command, keeping the training settings unchanged:

```bash
--resume-best runs/six/epoch_N.pth
```

Replace `N` with `best_epoch` from `checkpoints.json`.

### Optional Early Stopping

Early stopping is disabled by default. Enable it with `--early-stop-monitor`:

| Task | Monitor | Stop after no improvement |
|---|---|---|
| `binary` | `macro_precision_3mean`: mean image-level validation Macro-Precision over the last 3 epochs | 10 epochs |
| `six` | `fetal_macro_f1`: current fetal-level validation Macro-F1 | 15 epochs |

Binary diagnosis:

```bash
python train.py --task binary --train-list data/binary_train.txt --val-list data/binary_val.txt --output runs/binary_early --early-stop-monitor macro_precision_3mean --device cuda:0
```

Six-class diagnosis:

```bash
python train.py --task six --train-list data/six_train.txt --val-list data/six_val.txt --output runs/six_early --early-stop-monitor fetal_macro_f1 --val-case-map data/six_val_cases.jsonl --device cuda:0
```

For six-class early stopping, create `data/six_val_cases.jsonl` with one entry per validation image:

```json
{"image":"images/example_a_01.png","fetus_id":"example_a"}
{"image":"images/example_a_02.png","fetus_id":"example_a"}
{"image":"images/example_b_01.png","fetus_id":"example_b"}
```

Relative paths are resolved from the JSONL file's directory. Every validation image must appear exactly once. Labels come from `--val-list`, and all images of one fetus must share the same label. The validation set must contain every class.

The monitor selects the best checkpoint and restores it when training ends. Tied scores count as no improvement. The binary monitor starts checkpoint selection at epoch 3 and saves that epoch's weights, not averaged weights. See `early_stop_summary.json` for the stopping and restored epochs.

Keep `--epochs` at the intended maximum (default: 100); early stopping ends training automatically. These monitors are unavailable for `view` and cannot be combined with `--selection accuracy` or `--selection macro_f1`. To resume, keep the same monitor and data arguments and add `--resume-best` as shown above.

## Image Prediction and Evaluation

Predict a single image using the downloaded fixed weights:

```bash
python predict.py --task binary --image data/example.png --output results/example.json --device cuda:0
```

Evaluate a labeled test list using the training-validation preprocessing pipeline:

```bash
python predict.py --task six --list data/six_test.txt --protocol validation --output results/six_test.json --device cuda:0
```

To use your own trained weights, append:

```bash
--checkpoint runs/six/epoch_N.pth
```

Choose `view`, `binary`, or `six` for `--task`. The checkpoint must match the task. Without `--checkpoint`, the corresponding fixed model in `weights/` is used.

## Fetal-Level Inference

Create `data/fetal.jsonl` with one image and fetus ID per line:

```json
{"image":"images/example_001.png","fetus_id":"fetus_001","fetus_label":1}
{"image":"images/example_002.png","fetus_id":"fetus_001","fetus_label":1}
```

`fetus_label` is an optional fetal-level reference diagnosis.

```bash
# Binary diagnosis
python fetal.py --task binary --manifest data/fetal.jsonl --output results/fetal_binary.json --device cuda:0

# Six-class diagnosis
python fetal.py --task six --manifest data/fetal.jsonl --output results/fetal_six.json --device cuda:0
```

The pipeline screens for 4CH views and aggregates images from the same fetus into a fetal-level prediction.

## Grad-CAM

```bash
python gradcam.py --task six --image data/example.png --output results/cam --device cuda:0
```

The predicted class is visualized by default. Add `--target CLASS_INDEX` to select a class. Outputs include the preprocessed image, heatmap, and overlay.

## Tests

```bash
python -m tests.test_release
```

