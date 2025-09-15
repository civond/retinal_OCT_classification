import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


df1 = pd.read_csv("training_metrics_normal.csv")
df2 = pd.read_csv("training_metrics_qat.csv")

x1 = np.linspace(0, len(df1['training_acc']), len(df1['training_acc']))
x2 = np.linspace(0, len(df2['training_acc']), len(df2['training_acc']))

fig1, ax1 = plt.subplots(2, 2, figsize=(7, 6))
ax1[0,0].plot(x1, df1['training_loss'], color='b')
ax1[0,0].set_title("Training Loss")
ax1[0,0].set_ylabel("Cross Entropy Loss")
ax1[0,0].set_xlabel("Epoch")
ax1[0,0].set_xlim(0,len(df1['training_acc']))

ax1[0,1].plot(x1, df1['training_acc'], color='b')
ax1[0,1].set_title("Training Accuracy")
ax1[0,1].set_ylabel("Accuracy")
ax1[0,1].set_xlabel("Epoch")
ax1[0,1].set_xlim(0,len(df1['training_acc']))

ax1[1,0].plot(x1, df1['validation_loss'], color='r')
ax1[1,0].set_title("Validation Loss")
ax1[1,0].set_ylabel("Cross Entropy Loss")
ax1[1,0].set_xlabel("Epoch")
ax1[1,0].set_xlim(0,len(df1['training_acc']))

ax1[1,1].plot(x1, df1['validation_acc'], color='r')
ax1[1,1].set_title("Validation Accuracy")
ax1[1,1].set_ylabel("Accuracy")
ax1[1,1].set_xlabel("Epoch")
ax1[1,1].set_xlim(0,len(df1['training_acc']))
plt.tight_layout()
plt.savefig("normal_metrics.png")

# QAT 
fig2, ax2 = plt.subplots(2, 2, figsize=(7, 6))
ax2[0,0].plot(x2, df2['training_loss'], color='b', linestyle='--')
ax2[0,0].set_title("QAT Training Loss")
ax2[0,0].set_ylabel("Cross Entropy Loss")
ax2[0,0].set_xlabel("Epoch")
ax2[0,0].set_xlim(0,len(df2['training_acc']))

ax2[0,1].plot(x2, df2['training_acc'], color='b', linestyle='--')
ax2[0,1].set_title("QAT Training Accuracy")
ax2[0,1].set_ylabel("Accuracy")
ax2[0,1].set_xlabel("Epoch")
ax2[0,1].set_xlim(0,len(df2['training_acc']))

ax2[1,0].plot(x2, df2['validation_loss'], color='r', linestyle='--')
ax2[1,0].set_title("QAT Validation Loss")
ax2[1,0].set_ylabel("Cross Entropy Loss")
ax2[1,0].set_xlabel("Epoch")
ax2[1,0].set_xlim(0,len(df2['training_acc']))

ax2[1,1].plot(x2, df2['validation_acc'], color='r', linestyle='--')
ax2[1,1].set_title("QAT Validation Accuracy")
ax2[1,1].set_ylabel("Accuracy")
ax2[1,1].set_xlabel("Epoch")
ax2[1,1].set_xlim(0,len(df2['training_acc']))
plt.tight_layout()
plt.savefig("QAT_metrics.png")

plt.show()