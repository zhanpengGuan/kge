import torch

# 创建一个模型参数
weight = torch.randn(5, 5, requires_grad=True)

# 创建 emb_final，确保启用了梯度跟踪
emb_final = torch.randn(3, 5, requires_grad=True)

# 对权重进行赋值操作，使用 .clone() 创建新张量
indexes = torch.tensor([0, 2, 4])
new_values = emb_final.clone()  # 使用 .clone() 创建新张量
weight.data[indexes] = new_values.data  # 使用 .data 获取原始数据张量

# 计算损失并执行反向传播
loss = torch.sum(weight)
loss.backward()

# 查看 emb_final 的梯度
print(emb_final.grad)
