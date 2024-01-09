import torch

# 定义你的模型
checkpoint_path = '/home/guanzp/code/AdaE/kge/local/wnrr/auto/20240107-152247AdaE_auto-auto-cie--0.28--0.5-soft-512-drop-0.36small-gumbel/checkpoint_best.pt'

# 使用PyTorch加载检查点
checkpoint = torch.load(checkpoint_path)
emb = checkpoint['choice_emb']
Gpro =  torch.zeros(emb.shape,device='cuda:3')
choice = torch.argmax(emb, dim = -1)
# 打印参数
uni_choice = torch.unique(choice)

for i in uni_choice:
  print(i,sum(choice==i))
print(uni_choice)
