import pandas as pd
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

from utils.checkpoint import save_checkpoint
from utils.train_fn import train_fn
from utils.valid_fn import valid_fn
from utils.create_transforms import create_train_transform, create_valid_transform
from model import resnet101

learning_rate = 1e-4
batch_size = 64
num_epochs = 20
num_workers = 10
image_height = 224
image_width = 224
pin_memory = True
load_model = False
train = True

train_dir = "./data_train/"
valid_dir = "./data_valid/"
test_dir = "./data_test/"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    model = resnet101(num_classes=4)
    model.to(device)

    # Create train data loader
    train_transform = create_train_transform(image_height, image_width)
    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    train_loader = DataLoader(train_dataset, 
                            batch_size=batch_size, 
                            shuffle=True, 
                            num_workers=num_workers,
                            pin_memory=True)

    # Create validation data loader
    valid_transform = create_valid_transform(image_height, image_width)
    valid_dataset = datasets.ImageFolder(root=test_dir, transform=valid_transform)
    valid_loader = DataLoader(valid_dataset, 
                            batch_size=batch_size, 
                            shuffle=False, 
                            num_workers=num_workers,
                            pin_memory=True)

    # Loss function, optimizer and scaler.
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scaler = torch.amp.GradScaler(device=device)

    training_loss = []
    training_acc = []
    validation_loss = []
    validation_acc = []
    
    for epoch in range(num_epochs):
        print(f"\nEpoch: {epoch}")
        # Train
        train_loss, train_acc = train_fn(device, train_loader, model, optimizer, loss_fn, scaler)
        training_loss.append(train_loss)
        training_acc.append(train_acc)
        print(f"Train Loss: {train_loss}")
        print(f"Train Acc: {train_acc}")

        valid_loss, valid_acc = valid_fn(device, valid_loader, model, optimizer, loss_fn, scaler)
        validation_loss.append(valid_loss)
        validation_acc.append(valid_acc)
        print(f"Valid Loss: {valid_loss}")
        print(f"Valid Acc: {valid_acc}")

    # Save checkpoint
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    save_checkpoint(checkpoint)
    df = pd.DataFrame({
            "training_loss": training_loss,
            "training_acc": training_acc,
            "validation_loss": validation_loss,
            "validation_acc": validation_acc
        })
    df.to_csv("training_metrics_normal.csv", index=False)

if __name__ == "__main__":
    main()