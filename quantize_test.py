
from model import resnet101
from utils.checkpoint import load_checkpoint
import torch
import torch.nn as nn
import torch.optim as optim

from utils.checkpoint import load_checkpoint  # your checkpoint loader
from utils.create_transforms import create_train_transform, create_valid_transform
from utils.valid_fn import valid_fn
from torchvision import datasets
from torch.utils.data import DataLoader
from torchao.quantization import quantize_, Int8DynamicActivationInt4WeightConfig

"""model = resnet101(num_classes=4)
print(f"Base: {model.fc}")
base_config = Int8DynamicActivationInt4WeightConfig(group_size=32)
quantize_(model, QATConfig(base_config, step="prepare"))
print(f"Quant: {model.fc}")

# Convert
quantize_(model, QATConfig(base_config, step="convert"))
model.eval()
torch.save(model.state_dict(), 'test.pth')"""




learning_rate = 1e-4
batch_size = 64
num_epochs = 1
num_workers = 10
image_height = 224
image_width = 224
pin_memory = True
load_model = False
train = True

train_dir = "./data_train/"
valid_dir = "./data_valid/"
test_dir = "./data_test/"

def main():
    # Loading
    model = resnet101(num_classes=4).eval()
    state_dict = torch.load('QAT_model.pth')
    model.load_state_dict(state_dict, assign=True)
    model.eval()
    print(model.fc)

    """model.to("cpu")

    # Reset final FC layer. The model should be resilient to quantization error now.
    def reset_fc(model, num_classes):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)  # fresh FP32 layer
        return model
    model = reset_fc(model, num_classes=4)
    # Next, perform minor fine tuning to recover lost performance.
    #FINE TUNING SCRIPT)
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze only the last layer (fc)
    for param in model.fc.parameters():
        param.requires_grad = True

    # Then export as ONNX
    dummy_input = torch.randn(1, 1, 224, 224).to("cpu")  # adjust to your model’s input
    torch.onnx.export(
        model,
        dummy_input,
        "QAT_onnx.onnx",
        export_params=True,
        opset_version=17,   # recent opset for quantization
        do_constant_folding=True,
        dynamo=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )"""
    
    """# Create validation data loader
    valid_transform = create_valid_transform(image_height, image_width)
    valid_dataset = datasets.ImageFolder(root=test_dir, transform=valid_transform)
    valid_loader = DataLoader(valid_dataset, 
                            batch_size=batch_size, 
                            shuffle=False, 
                            num_workers=num_workers,
                            pin_memory=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scaler = torch.amp.GradScaler(device=device)

    valid_loss, valid_acc = valid_fn(device, valid_loader, model, optimizer, loss_fn, scaler)

    print(f"Valid Loss: {valid_loss}")
    print(f"Valid Acc: {valid_acc}")
    """
if __name__ == "__main__":
    main()