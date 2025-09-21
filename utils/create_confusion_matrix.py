import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

# Load CSV
df = pd.read_csv("./csv/inference_results_int8.csv")

# Extract actual & predicted labels
y_true = df["actual_label"]
y_pred = df["predicted_label"]

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Compute accuracy
acc = accuracy_score(y_true, y_pred)
print(f"Accuracy: {acc:.4f}")

# Class names
class_names = ["CNV", "DME", "DRUSEN", "NORMAL"]

# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title(f"QAT Inference (acc={acc:.2f})")
plt.tight_layout()
plt.show()