# 需要移植的模块
  ## embedder
  - 在kge_model.py中
    ```python
    s = self.get_s_embedder().embed(s)
    ```
    这里是通过s（索引）取得embedding的地方，因此直接写一个新的函数
    首先，重写embedder，得到multi-embedder
    - multi-embedder.py
    然后重写embed_all()和embed(),
    在_adaE()里面写picker+selecter
    补齐所有的参数
      train_mode代表的模块 fix,original写完了
    23/0906
    adaE_init()包含了multi_embedding的初始化和dropout
    ！！ 初始化我用的是parameters，还有库里的initinize,可能有问题
    写到rank_e了，这里需要对Dataset下手了,最终写了方法count_entity_frequency，在multi_lookup_embedder里调用。
    [0.2]意味着最高的20%和低频的80%的划分



  ## picker
  - AdaE.py
  这个和embedder分离，写了一个类

  - optimizer
  - 双层规划的模块
  ## 解析器argparse
    
  Class Config.py里引入了config-default.yaml作为合法参数,_import()函数会引入所有model.yaml作为合法参数
  因此直接在config-default.yaml中加入需要的参数即可
  这里共分为三步
  - 写新模型和模型对应的yaml（此处不仅要改class name,还要改最前面的那个类名字）
  - init里面import他们 
  - test_config.yaml里面需要import它们


  - save模块
  - Dataset/训练集的数据分割
  - 测试模块
  - KgeSampler采样任务
# logic architectrue
  - job
    - Training job
      - Train_negative_sampling job
  - kgemodel
    - reciprcal_realtions_model
      - AdaE
      
  -kge_embedder
    - lookup_embedder
  - scorer
  - loss
# 一些重要的类函数
  - train.py
    run_epoch
    _process_batch
  - train_negative_sampling.py
    _process_subbatch
  - kge_model.py
    score_spo, spo都是大小为[batch]
# 遇到的bug和解决方案
  ## 1
  在AdaE这个子类中重写get_s_embedder，要注意此时AdaE没有entity_embedder，只有multi_entity_embedder
  在AdaE的父类Reciprocal_relation_model中，只有entity_embedder，这个来自于base_model:AdaE的_multi_entity_embedder
  ## 2
  RoTaTE的初始化沒寫，可能會報錯
  ## 3
  必须注意，当出现新的choice_list,模型的count_e_entity的函数里，用于加速的加载已存在的rank_e必须要调整。
