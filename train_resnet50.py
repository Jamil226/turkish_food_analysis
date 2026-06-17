import os
import csv
import torch
import torch.nn as nn
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = "DejaVu Sans"

from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

# =====================================================
# CONFIG
# =====================================================

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_DIR = os.path.join(PROJECT_DIR, "data", "train")
VAL_DIR   = os.path.join(PROJECT_DIR, "data", "val")

CHECKPOINT_DIR  = os.path.join(PROJECT_DIR, "checkpoints", "resnet50")
OUTPUT_DIR      = os.path.join(PROJECT_DIR, "outputs", "resnet50")
BEST_MODEL_PATH = os.path.join(PROJECT_DIR, "best_models", "resnet50_best.pth")

# Two-stage training: freeze backbone first, then unfreeze layer4 + fc
STAGE1_EPOCHS = 10   # frozen backbone — train fc only
STAGE2_EPOCHS = 10   # fine-tune layer4 + fc
BATCH_SIZE    = 128
LR_STAGE1     = 1e-3
LR_STAGE2     = 1e-4

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(BEST_MODEL_PATH), exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =====================================================
# TRANSFORMS
# =====================================================

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(15),
    transforms.ColorJitter(
        brightness=0.3,
        contrast=0.3,
        saturation=0.3,
        hue=0.1,
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    ),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
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

model = models.resnet50(weights="IMAGENET1K_V1")

# Stage 1: freeze entire backbone
for param in model.parameters():
    param.requires_grad = False

# Replace classifier head for num_classes
model.fc = nn.Linear(model.fc.in_features, num_classes)

model = model.to(device)

# =====================================================
# TRAINING SETUP
# =====================================================

criterion = nn.CrossEntropyLoss()
scaler    = torch.amp.GradScaler("cuda")

# =====================================================
# TRAIN FUNCTION
# =====================================================

def train_one_epoch(optimizer):
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
            loss    = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        preds       = outputs.argmax(1)
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

    true_labels  = []
    pred_labels  = []
    image_paths  = []

    with torch.no_grad():
        image_index = 0

        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss    = criterion(outputs, labels)

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
# LOGGING SETUP
# =====================================================

best_accuracy = 0.0

log_file = os.path.join(OUTPUT_DIR, "training_log.csv")

if not os.path.exists(log_file):
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "stage", "train_loss", "train_acc", "val_loss", "val_acc"])


# =====================================================
# STAGE 1: TRAIN FC ONLY (frozen backbone)
# =====================================================

print("\n" + "=" * 50)
print("STAGE 1: Training classifier head (frozen backbone)")
print("=" * 50)

optimizer_s1 = torch.optim.AdamW(model.fc.parameters(), lr=LR_STAGE1)
scheduler_s1 = torch.optim.lr_scheduler.StepLR(optimizer_s1, step_size=5, gamma=0.1)

for epoch in range(STAGE1_EPOCHS):
    print(f"\nEpoch {epoch + 1}/{STAGE1_EPOCHS}  [Stage 1]")

    train_loss, train_acc = train_one_epoch(optimizer_s1)
    val_loss, val_acc, true_labels, pred_labels, image_paths = validate()

    print(f"Train Loss: {train_loss:.4f}")
    print(f"Train Acc : {train_acc:.2f}%")
    print(f"Val Loss  : {val_loss:.4f}")
    print(f"Val Acc   : {val_acc:.2f}%")

    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch + 1, "stage1", train_loss, train_acc, val_loss, val_acc])

    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"stage1_epoch_{epoch + 1:03d}.pth")
    torch.save(
        {
            "epoch": epoch,
            "stage": "stage1",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer_s1.state_dict(),
        },
        checkpoint_path,
    )

    if val_acc > best_accuracy:
        best_accuracy = val_acc
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"Best model saved ({best_accuracy:.2f}%) → {BEST_MODEL_PATH}")

    scheduler_s1.step()

print(f"\nStage 1 best accuracy: {best_accuracy:.2f}%")

# =====================================================
# STAGE 2: FINE-TUNE layer4 + fc
# =====================================================

print("\n" + "=" * 50)
print("STAGE 2: Fine-tuning layer4 + classifier head")
print("=" * 50)

# Load best stage-1 model before fine-tuning
model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))

# Unfreeze layer4 and fc
for name, param in model.named_parameters():
    if "layer4" in name or "fc" in name:
        param.requires_grad = True

optimizer_s2 = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR_STAGE2,
)
scheduler_s2 = torch.optim.lr_scheduler.StepLR(optimizer_s2, step_size=5, gamma=0.1)

for epoch in range(STAGE2_EPOCHS):
    global_epoch = STAGE1_EPOCHS + epoch + 1
    print(f"\nEpoch {epoch + 1}/{STAGE2_EPOCHS}  [Stage 2  |  Global epoch {global_epoch}]")

    train_loss, train_acc = train_one_epoch(optimizer_s2)
    val_loss, val_acc, true_labels, pred_labels, image_paths = validate()

    print(f"Train Loss: {train_loss:.4f}")
    print(f"Train Acc : {train_acc:.2f}%")
    print(f"Val Loss  : {val_loss:.4f}")
    print(f"Val Acc   : {val_acc:.2f}%")

    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([global_epoch, "stage2", train_loss, train_acc, val_loss, val_acc])

    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"stage2_epoch_{epoch + 1:03d}.pth")
    torch.save(
        {
            "epoch": global_epoch,
            "stage": "stage2",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer_s2.state_dict(),
        },
        checkpoint_path,
    )

    if val_acc > best_accuracy:
        best_accuracy = val_acc
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"Best model saved ({best_accuracy:.2f}%) → {BEST_MODEL_PATH}")

    scheduler_s2.step()

print(f"\nStage 2 best accuracy: {best_accuracy:.2f}%")

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
