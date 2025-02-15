import torch

# 检查是否有CUDA可用，如果有，则使用CUDA加速
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载result.pt文件
path = []
path.append('data/fb15k-237/[-1]rank_e.pt')
path.append("data/wnrr/[-1]rank_e.pt")
path.append('data/yago3-10/[-1]rank_e.pt')
for i in range(3):

    data = torch.load(path[i], map_location=device)

    data = data.float()
 

    # 计算均值
    mean = torch.mean(data)

    # 计算方差
    variance = torch.var(data)

    # 计算标准差
    std_deviation = torch.std(data)

    # 标准化数据
    normalized_data = (data - mean) / std_deviation

    # 计算标准化后的方差
    normalized_variance = torch.var(normalized_data)
    print(path[i].split('/')[1])
    print("均值:", mean)
    print("方差:", variance)
    print("标准差:", std_deviation)


