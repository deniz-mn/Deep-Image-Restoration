# Deep Image Restoration

## Overview

A computer vision course group project exploring how supervised deep learning can restore damaged photographs affected by noise, scratches, and faded colors. The project implements and evaluates a PyTorch U-Net that maps damaged images to paired clean targets.

## ✨ Key Highlights

- Implemented a four-stage U-Net with skip connections, batch normalization, transposed convolutions, and approximately 31 million parameters.
- Built a paired-image pipeline using `joshuachin/openphoto-restore-dataset`: RGB conversion, 256×256 resizing, normalization, and seeded train/validation splitting.
- Trained for 30 epochs with L1 loss and Adam, saving the checkpoint with the lowest validation loss.
- Evaluated 500 held-out images using four metrics and generated loss curves and side-by-side restoration examples.

## 🛠️ Tech Stack

Python · PyTorch · torchvision · Hugging Face Datasets · NumPy · pandas · scikit-image · Matplotlib · tqdm · Jupyter/Google Colab · CUDA

## ⚙️ How It Works / Usage

From the repository directory, install dependencies and open the notebook:

```bash
python -m pip install torch torchvision datasets hf_xet numpy pillow pandas scikit-image matplotlib tqdm jupyterlab
python -m jupyterlab "CV_Final_Project_Restoration_(2).ipynb"
```

Run cells in order: download data → preprocess → train → reload the best checkpoint → evaluate and visualize. Internet access is needed for the dataset; the notebook selects CUDA when available and otherwise uses CPU.

Training creates `checkpoints/unet_best.pt`; plots and metrics are written to `outputs/`. Trained weights are not included, so evaluation requires completing training first.

## 🧠 Skills Demonstrated

- Custom PyTorch datasets, encoder–decoder design, and training/validation loops.
- Checkpoint selection and evaluation on a separate test set.
- Comparing numerical image-quality metrics with visual results, including limitations in color recovery.

## 📊 Results

Recorded test-set averages from [test_metrics.csv](test_metrics.csv):

| PSNR ↑ | SSIM ↑ | MSE ↓ | MAE ↓ |
| --- | --- | --- | --- |
| 23.00 dB | 0.8626 | 0.007951 | 0.057866 |

Both course targets were met: PSNR ≥ 20 dB and SSIM ≥ 0.65. Sample outputs retain muted colors, showing that structural restoration does not guarantee accurate color recovery.

[Restoration examples](sample_restorations.png) · [Loss curve](loss_curve.png) · [Project report](CV_Final_Project_Report.pdf)
