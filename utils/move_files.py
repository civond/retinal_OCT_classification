import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

# Path to your dataset (with NORMAL, CNV, DRUSEN, DME inside)
root_dir = Path("./data")

# Output directory
output_dir = Path("dataset_split")
for split in ["train", "val", "test"]:
    for cls in ["NORMAL", "CNV", "DRUSEN", "DME"]:
        (output_dir / split / cls).mkdir(parents=True, exist_ok=True)

# Collect files + labels
all_files = []
all_labels = []
for cls in ["NORMAL", "CNV", "DRUSEN", "DME"]:
    class_dir = root_dir / cls
    files = list(class_dir.glob("*"))
    all_files.extend(files)
    all_labels.extend([cls] * len(files))

# First split: train (80%) and temp (20%)
train_files, temp_files, train_labels, temp_labels = train_test_split(
    all_files, all_labels, test_size=0.2, stratify=all_labels, random_state=42
)

# Second split: val (10%) and test (10%)
val_files, test_files, val_labels, test_labels = train_test_split(
    temp_files, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
)

# Helper function to copy
def copy_files(files, labels, split):
    for f, lbl in zip(files, labels):
        shutil.copy(f, output_dir / split / lbl / f.name)

copy_files(train_files, train_labels, "train")
copy_files(val_files, val_labels, "val")
copy_files(test_files, test_labels, "test")

print("✅ Dataset successfully split (80/10/10) with sklearn.train_test_split")
