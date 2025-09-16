import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


df1 = pd.read_csv("training_metrics_normal.csv")
df2 = pd.read_csv("training_metrics_qat.csv")
x1 = np.linspace(0, len(df1['training_acc'])-1, len(df1['training_acc']))
x2 = np.linspace(0, len(df2['training_acc'])-1, len(df2['training_acc']))

fig1, ax1 = plt.subplots(2, 2, figsize=(7, 6))

# Training Loss
ax1[0,0].plot(x1, df1['training_loss'], 
              color='royalblue', 
              linestyle='-', 
              linewidth=2, 
              label='normal')
ax1[0,0].plot(x2, df2['training_loss'], 
              color='crimson', 
              linestyle='--', 
              linewidth=2, 
              label='QAT')
ax1[0,0].set_title("Training Loss")
ax1[0,0].set_ylabel("Cross Entropy Loss")
ax1[0,0].set_xlabel("Epoch")
ax1[0,0].set_xlim(0,len(df1['training_acc'])-1)
ax1[0,0].legend()

# Training Accuracy
ax1[0,1].plot(x1, df1['training_acc'], 
              color='royalblue', 
              linestyle='-', 
              linewidth=2, 
              label='normal')
ax1[0,1].plot(x2, df2['training_acc'], 
              color='crimson', 
              linestyle='--', 
              linewidth=2, 
              label='QAT')
ax1[0,1].set_title("Training Accuracy")
ax1[0,1].set_ylabel("Accuracy")
ax1[0,1].set_xlabel("Epoch")
ax1[0,1].set_xlim(0,len(df1['training_acc'])-1)
ax1[0,1].legend()

# Validation Loss
ax1[1,0].plot(x1, df1['validation_loss'], 
              color='royalblue', 
              linestyle='-', 
              linewidth=2, 
              label='normal')
ax1[1,0].plot(x2, df2['validation_loss'], 
              color='crimson', 
              linestyle='--', 
              linewidth=2, 
              label='QAT')
ax1[1,0].set_title("Validation Loss")
ax1[1,0].set_ylabel("Cross Entropy Loss")
ax1[1,0].set_xlabel("Epoch")
ax1[1,0].set_xlim(0,len(df1['training_acc'])-1)
ax1[1,0].legend()

# Validation Accuracy
ax1[1,1].plot(x1, df1['validation_acc'], 
              color='royalblue', 
              linestyle='-', 
              linewidth=2, 
              label='normal')
ax1[1,1].plot(x2, df2['validation_acc'], 
              color='crimson', 
              linestyle='--', 
              linewidth=2, 
              label='QAT')
ax1[1,1].set_title("Validation Accuracy")
ax1[1,1].set_ylabel("Accuracy")
ax1[1,1].set_xlabel("Epoch")
ax1[1,1].set_xlim(0,len(df1['training_acc'])-1)
ax1[1,1].legend()

plt.tight_layout()
plt.savefig("normal_metrics.png")
plt.show()