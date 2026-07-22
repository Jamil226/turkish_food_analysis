# Turkish Food Analysis

This repository contains the food-classification part of the study described in paper entitled, **A Hybrid CNN–MLLM Architecture for Image-Based Nutrition Estimation and Advisory Insulin Decision Support in Type 1 Diabetes**.

The code in this folder trains and evaluates three CNN backbones for Turkish food image classification:

- ResNet50
- Inception V3
- EfficientNet-B0

The paper reports the classification model as the first stage of a larger nutrition-estimation and insulin decision-support pipeline. In this repository, the implemented workflow focuses on image classification, evaluation, and result reporting.

## Repository Layout

- `train_resnet50.py` - two-stage training and evaluation for ResNet50
- `train_inception.py` - training and evaluation for Inception V3
- `train_efficientnet.py` - training and evaluation for EfficientNet-B0
- `generate_report.py` - converts `predictions.csv` into a percentage-based classification report
- `plot_confusion_matrix.py` - creates a high-quality confusion matrix image from `predictions.csv`
- `best_models/` - best saved weights for each backbone
- `outputs/<model>/` - per-model logs, predictions, reports, and plots
- `paper.pdf` - manuscript describing the study and reported results

## What The Code Does

Each training script:

- loads images from `data/train` and `data/val`
- applies ImageNet-style normalization and model-specific augmentation
- trains on GPU if CUDA is available, otherwise falls back to CPU
- saves periodic checkpoints under `checkpoints/<model>/`
- stores the best model weights under `best_models/<model>_best.pth`
- writes validation predictions to `outputs/<model>/predictions.csv`
- writes a plain-text classification report to `outputs/<model>/classification_report.txt`
- writes a confusion matrix image to `outputs/<model>/confusion_matrix.png`

Model-specific details:

- `train_resnet50.py` uses a two-stage schedule: freeze the backbone first, then fine-tune `layer4` and `fc`
- `train_inception.py` uses 299 x 299 inputs and includes the auxiliary classifier loss during training
- `train_efficientnet.py` uses EfficientNet-B0 with a custom classifier head

## Requirements

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

The main libraries used are PyTorch, torchvision, pandas, scikit-learn, seaborn, matplotlib, numpy, and Pillow.

## Data Setup

The training scripts expect an ImageFolder-style dataset structure:

```text
data/
  train/
    class_1/
    class_2/
    ...
  val/
    class_1/
    class_2/
    ...
```

The class folder names become the label names used during training and evaluation.

## Dataset Access

The dataset used in this study is not publicly available. Access can be requested through institutional email. Here is the contact email:

- jamil138.amin@gmail.com

## Training

Run one model at a time:

```bash
python train_resnet50.py
python train_inception.py
python train_efficientnet.py
```

Each script automatically creates the needed output folders if they do not already exist.

## Reporting And Plots

After a training script finishes, you can regenerate the formatted report and confusion matrix for a given model:

```bash
python generate_report.py --model efficientnet
python plot_confusion_matrix.py --model efficientnet
```

Replace `efficientnet` with `inception` or `resnet50` as needed.

## Outputs

For each model, the repository stores results under `outputs/<model>/`:

- `training_log.csv`
- `predictions.csv`
- `classification_report.txt`
- `classification_report_pct.txt`
- `confusion_matrix.png`
- `confusion_matrix_hq.png`

The `best_models/` directory contains the best-performing weights:

- `best_models/resnet50_best.pth`
- `best_models/inception_best.pth`
- `best_models/efficientnet_best.pth`

## Paper Summary

The manuscript frames the work as an AI-assisted diabetes-care pipeline that estimates meal composition from food images and supports advisory insulin decisions. The evaluated classification benchmark uses 40 food categories and compares three CNN architectures. According to the paper, EfficientNet-B0 achieved the best validation performance among the evaluated models.

## Notes

- The training scripts use mixed-precision CUDA training when a GPU is available.
- `train_inception.py` expects a pretrained Inception V3 backbone and optionally loads `pretrained/inception_v3_food_classification.pth` if present.
- `train_efficientnet.py` optionally loads `pretrained/efficientnet_food_classification.pth` if present.
- `train_resnet50.py` performs stage-based fine-tuning and saves the best stage-1 and stage-2 weights to `best_models/resnet50_best.pth`.

## Reproducibility

The exact results reported in the paper depend on the dataset split, image preprocessing, and the availability of pretrained weights. If you retrain the models, keep the train/validation folder structure and class names consistent across runs.

## Citation

If you find this repository or our research helpful in your work, please consider citing our paper:

### IEEE Style:

J. C. Velombe, S. Bayraktar, A. Kavak, M. Jamil, A. B. İçnner, G. Srivastava, and H. Fotouhi, "A Hybrid CNN–MLLM Architecture for Image-Based Nutrition Estimation and Advisory Insulin Decision Support in Type 1 Diabetes," Nutrients, vol. 18, no. 13, p. 2205, 2026.

### bibtex:

```bibtex
@article{velombe2026hybrid,
  title={A Hybrid CNN--MLLM Architecture for Image-Based Nutrition Estimation and Advisory Insulin Decision Support in Type 1 Diabetes},
  author={Velombe, Jean Chrinot and Bayraktar, Sema and Kavak, Adnan and Jamil, Muhammad and {\.I}nner, Alpaslan Burak and Srivastava, Gautam and Fotouhi, Hossein},
  journal={Nutrients},
  volume={18},
  number={13},
  pages={2205},
  year={2026},
  publisher={MDPI}
}
```

We appreciate your support and welcome citations of our work if it contributes to your research.
