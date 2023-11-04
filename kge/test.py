import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from collections import Counter
import torch.nn as nn
import torch.nn.functional as F
checkpoint_file = "data/fb15k-237/[-1]rank_e[-1].pt"
# 创建一个模型参数
# my_list = torch.load(checkpoint_file, map_location="cpu")
# x = {}
# for i in my_list:
#   j =int(i.item())
#   if j not in x:
#     x[j] = 1
#   else:
#     x[j]+=1

# data = list(x.keys())
# # 计算均值和标准差
# mean = np.mean(data)
# std_dev = np.std(data)

# # 使用z-score标准化数据
# standardized_data = stats.zscore(data)

# print("原始数据:", data)
# print("标准化后的数据:", standardized_data)


# # 提取元素和对应的计数
# elements = list(x.keys())
# counts = list(x.values())

# # 创建条形图
# plt.bar(elements, counts)
# plt.xlabel("元素")
# plt.ylabel("数量")
# plt.title("元素重复数量")
# plt.show()
pro = torch.ones(10,12)
pro[:,-1] = 5
for t in [0.0001,0.01,0.1,1,10,100,1000,10000]:
  Gpro = F.gumbel_softmax(pro, tau=t, hard=False)
  ans = 0
  for num in range(pro.shape[0]):
      if torch.argmax(Gpro[num])==torch.argmax(pro[num]):
          ans+=1
  print(ans)

  
