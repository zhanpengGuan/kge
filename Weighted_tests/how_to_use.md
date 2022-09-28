# Weighted KG
本库基于libkge进行改写，更多详细用法请看
https://github.com/uma-pi1/kge
## 目录
- how to use
- how to modify
## how to use
### 安装环境、下载数据
(如果已经安装，请跳过此小节)
```sh
# retrieve and install project in development mode
git clone xxx
cd kge
pip install -e .

# download and preprocess datasets
cd data
sh download_all.sh
cd ..

# train an example model on toy dataset (you can omit '--job.device cpu' when you have a gpu)
kge start examples/toy-complex-train.yaml --job.device cpu
```
### 运行Weighted KG model
 - 简要介绍
   - Weighted KG model的配置yaml文件存Weighted_tests/Weighted/ 文件夹下
   - yaml名为：W_model_data.yaml
   model:[transe,distmult,conve,complex,rescal]
   data:{ f: fb15-237k, w: wn18rr}
 - 运行
  ```sh
  kge start Weighted_tests/Weighted/XXX.yaml
  ```
- 可选设备 
在运行的命令行后添加--job.device cuda:0即可
### yaml 参数介绍
相较于libkge，新增了如下参数
 - train.type: 
   - [W_1vsAll,W_negitive_sampling]
 - model(e.g. W_transe).clip:
   - Weight上限
 - model.init: 
   - False：Weight初始化为1
   - True: Weight初始化为frequency based
 - model.require_grad_relation: 
   - True： relation的权重进行更新
   - False：relation的权重固定
 - model.require_grad_entity: 
   - True： entity的权重进行更新
   - False：entity的权重固定
 - model.golden_ratio: 
   - 0.2 高低频分割比例为降序20%以内为高频词，否则为低频词
### grid search
- 按照原来的方式写