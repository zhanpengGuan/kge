import torch
import torchvision.models as models

# Initialize your model (e.g., a pre-trained ResNet model)
model = models.resnet18()

# Load the checkpoint
checkpoint = torch.load('/data1/gzp/local/fb15k-237/auto/20231230-144141AdaE_auto-auto-cie--0.28--0.5-soft-512-drop-0.5-frequency/checkpoint_best.pt')

print(checkpoint)
print("final!")

## 目前的问题应该是AdaE在当时跑的时候，实现了一个非relation2倍的效果