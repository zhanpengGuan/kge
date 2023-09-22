# 需要移植的模块
  ## embedder
  此处设计的aligment模块包括在multi_embedder之中
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
    23/0907
    记得要初始化_embeddings
    20/0910
    正则化函数需要重写？此处_embeddings不能被更新，但要作为中间变量传递梯度。有两个解决办法
    第一，跟RotatE一样，重写Penty函数。
    第二，_embeddings也作为参数，但是optimizer不包括它。用了这个方法
    train_mode代表的模块 fix,original，rank写完了需要实验调试证明没啥问题
    23/0913
    目前看起来要写Darts的过程，总体设计如下
    重写一个继承train_negative_sample.job的类traindarts，用于batch的处理和architecture
      - 重写run_epoch()方法
      处理batch的时候有两种选择：
      - 在collate中处理，后续没有任何操作，但这样无法处理s_u=2的情况？（也可以处理，需要）
      - 在后续处理，可能涉及到label和batch,neg_sample的划分，这些比较麻烦
    在_adaE的方法里加上auto，也就是picker产生可变化概率的处理，aligment加上gumbel softmax
    写到了architectrue了,需要写optimizer_c和picker的参数。
    2023/0918
    架构方面，model包含embedder等参数，job则包含optimizer等以及训练方式
    不需要重写optimizer，只选用用新的optimizer包含picker的参数即可
    2023/09/18
    正在传递picker,目前的想法是在AdaE里初始化picker，然后再embedder初始化传入picker：*此想法不可行，因为不可以再初始化之前先初始化picker这个类，而且传入需要新增参数破坏了源代码这个*
    2023/09/20
    
    

    
    

  ## picker
  - AdaE.py
  这个和embedder分离，写了一个类Picker,用来包自己的参数。bucket,FC,softmax,gumbel_softmax.

  # optimizer
    需要加入optimizer_p
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

  - device需要设计
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
    _prepare():初始化了dataset，这里预处理数据集的方法都写在这个文件中
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
  ## traindarts.py
  batch在划分的时候，选取的shape是tripels的，可能会在其他模式里出现问题，在negsample没问题
  ## multi_embedder.py
  有一步没写，就是emb_all()