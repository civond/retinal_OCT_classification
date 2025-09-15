import torch
from torchao.quantization import quantize_, Int8DynamicActivationInt4WeightConfig
from torchao.quantization.qat import QATConfig

def save_checkpoint(state, filename = "my_checkpoint.pth.tar"):
    print("-> Saving checkpoint.")
    torch.save(state, filename)

def load_checkpoint(checkpoint_path, model):
    print("-> Loading checkpoint.")
    base_config = Int8DynamicActivationInt4WeightConfig(group_size=32)
    quantize_(model, QATConfig(base_config, step="convert"))
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    #checkpoint = torch.load(checkpoint_path, map_location="gpu")
    model.load_state_dict(checkpoint["state_dict"])