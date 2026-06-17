import os
import csv
import torch
import torch.nn as nn
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = "DejaVu Sans"

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import inception_v3, Inception_V3_Weights
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

# =====================================================
# CONFIG
# =====================================================

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_DIR = os.path.join(PROJECT_DIR, "data", "train")
VAL_DIR   = os.path.join(PROJECT_DIR, "data", "val")

CHECKPOINT_DIR  = os.path.join(PROJECT_DIR, "checkpoints", "inception")
OUTPUT_DIR      = os.path.join(PROJECT_DIR, "outputs", "inception")
BEST_MODEL_PATH = os.path.join(PROJECT_DIR, "best_models", "inception_best.pth")

PRETRAINED_MODEL = os.path.join(
    PROJECT_DIR, "pretrained", "inception_v3_food_classification.pth"
)

NUM_EPOCHS    = 20
BATCH_SIZE    = 128
LEARNING_RATE = 1e-3

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(BEST_MODEL_PATH), exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =====================================================
# TRANSFORMS
# Inception V3 requires 299x299 input (not 224x224)
# =====================================================

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(299),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1,
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    ),
])

val_transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    ),
])

# =====================================================
# DATASETS
# =====================================================

train_dataset = ImageFolder(TRAIN_DIR, transform=train_transform)
val_dataset   = ImageFolder(VAL_DIR,   transform=val_transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
)

num_classes = len(train_dataset.classes)
print("Classes:", train_dataset.classes)
print("Number of classes:", num_classes)

# =====================================================
# MODEL
# =====================================================

model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)

# Replace both the main classifier and the auxiliary classifier
model.fc     = nn.Linear(model.fc.in_features, num_classes)
model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, num_classes)

if os.path.exists(PRETRAINED_MODEL):
    print("Loading pretrained model...")
    checkpoint = torch.load(PRETRAINED_MODEL, map_location=device)
    # Filter out classifier keys that may not match num_classes
    checkpoint = {
        k: v for k, v in checkpoint.items()
        if not k.startswith("fc") and not k.startswith("AuxLogits.fc")
    }
    missing, unexpected = model.load_state_dict(checkpoint, strict=False)
    print(f"  Loaded backbone. Skipped keys: {missing}")

model = model.to(device)

# =====================================================
# TRAINING SETUP
# =====================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

scaler = torch.amp.GradScaler("cuda")

# =====================================================
# TRAIN FUNCTION
# Inception V3 returns (logits, aux_logits) during training.
# Auxiliary loss weighted by 0.4 as per the original paper.
# =====================================================

def train_one_epoch():
    model.train()

    total_loss = 0
    correct    = 0
    total      = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):
            outputs = model(images)

            # During training Inception returns InceptionOutputs(logits, aux_logits)
            if isinstance(outputs, tuple):
                logits, aux_logits = outputs
                loss = criterion(logits, labels) + 0.4 * criterion(aux_logits, labels)
            else:
                logits = outputs
                loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        preds       = logits.argmax(1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)

    return total_loss / len(train_loader), 100 * correct / total


# =====================================================
# VALIDATION
# =====================================================

def validate():
    model.eval()

    total_loss = 0
    correct    = 0
    total      = 0

    true_labels = []
    pred_labels = []
    image_paths = []

    with torch.no_grad():
        image_index = 0

        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            # During eval Inception returns plain logits
            if isinstance(outputs, tuple):
                outputs, _ = outputs

            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds       = outputs.argmax(1)
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)

            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(preds.cpu().numpy())

            batch_size = labels.size(0)
            for i in range(batch_size):
                image_paths.append(val_dataset.samples[image_index + i][0])

            image_index += batch_size

    accuracy = 100 * correct / total

    return (
        total_loss / len(val_loader),
        accuracy,
        true_labels,
        pred_labels,
        image_paths,
    )


# =====================================================
# TRAIN LOOP
# =====================================================

best_accuracy = 0.0

log_file = os.path.join(OUTPUT_DIR, "training_log.csv")

if not os.path.exists(log_file):
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")

    train_loss, train_acc = train_one_epoch()
    val_loss, val_acc, true_labels, pred_labels, image_paths = validate()

    print(f"Train Loss: {train_loss:.4f}")
    print(f"Train Acc : {train_acc:.2f}%")
    print(f"Val Loss  : {val_loss:.4f}")
    print(f"Val Acc   : {val_acc:.2f}%")

    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch + 1, train_loss, train_acc, val_loss, val_acc])

    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch + 1:03d}.pth")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        checkpoint_path,
    )

    if val_acc > best_accuracy:
        best_accuracy = val_acc
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"Best model saved ({best_accuracy:.2f}%) → {BEST_MODEL_PATH}")

    scheduler.step()

# =====================================================
# FINAL EVALUATION
# =====================================================

print("\nLoading best model...")
model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))

val_loss, val_acc, true_labels, pred_labels, image_paths = validate()
print(f"\nFinal Validation Accuracy: {val_acc:.2f}%")

report = classification_report(
    true_labels,
    pred_labels,
    target_names=train_dataset.classes,
)
print(report)

with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
    f.write(report)

pred_df = pd.DataFrame({
    "image_path": image_paths,
    "true_class": [train_dataset.classes[x] for x in true_labels],
    "predicted_class": [train_dataset.classes[x] for x in pred_labels],
})

pred_df.to_csv(
    os.path.join(OUTPUT_DIR, "predictions.csv"),
    index=False,
)

cm = confusion_matrix(true_labels, pred_labels)

plt.figure(figsize=(15, 15))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=train_dataset.classes,
    yticklabels=train_dataset.classes,
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))

print("\nDone.")
