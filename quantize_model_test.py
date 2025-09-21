
import torch
from torchao.quantization import quantize_, Int8DynamicActivationInt4WeightConfig, Int8WeightOnlyConfig
from torchao.quantization.qat import QATConfig
import os
import torchao.quantization as ao

from model import resnet101
from torchao.quantization import Int4WeightOnlyConfig, quantize_

#model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)

model = resnet101(num_classes=4).eval()
state_dict = torch.load("QAT_model.pth", map_location="cpu")
model.load_state_dict(state_dict, assign=True)
torch.save(model, "./tmp/fp32_model.pt")

# Apply int8 quantization
config = Int8WeightOnlyConfig(group_size=32)
quantize_(model, config)

torch.save(model, "./tmp/fp32_model.pt")

#config = Int8DynamicActivationInt4WeightConfig(group_size=32)
#config = Int8WeightOnlyConfig(group_size=32)
#quantize_(model, config)
torch.save(model, "./tmp/int8_model.pt")

int4_model_size_mb = os.path.getsize("./tmp/fp32_model.pt") / 1024 / 1024
bfloat16_model_size_mb = os.path.getsize("./tmp/int8_model.pt") / 1024 / 1024
print("fp32 model size: %.2f MB" % int4_model_size_mb)
print("int8 model size: %.2f MB" % bfloat16_model_size_mb)



"""torch.save(model, "int8_model.pt")
torch.save(model.state_dict(), "int8_model_state.pth")"""



##### This example works???
"""class ToyLinearModel(torch.nn.Module):
    def __init__(self, m: int, n: int, k: int):
        super().__init__()
        self.linear1 = torch.nn.Linear(m, n, bias=False)
        self.linear2 = torch.nn.Linear(n, k, bias=False)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x"""


"""torch.save(model, "./tmp/float16_model.pt")
from torchao.quantization import Int4WeightOnlyConfig, quantize_
quantize_(model, Int8WeightOnlyConfig(group_size=32))


import os
torch.save(model, "./tmp/int4_model.pt")
#torch.save(model_bf16, "/tmp/bfloat16_model.pt")
int4_model_size_mb = os.path.getsize("./tmp/int4_model.pt") / 1024 / 1024
bfloat16_model_size_mb = os.path.getsize("./tmp/float16_model.pt") / 1024 / 1024

print("int4 model size: %.2f MB" % int4_model_size_mb)
print("bfloat16 model size: %.2f MB" % bfloat16_model_size_mb)"""