import torch
checkpoint_file = "local/experiments/fb15k-237/20231015-180404-AdaE_auto-auto-noshare-[0.2]-[256, 512]-ts-LN-1vsall--512-0.5-0.5"
# 创建一个模型参数
checkpoint = torch.load(checkpoint_file+"/checkpoint_best.pt", map_location="cpu")
print(checkpoint)
