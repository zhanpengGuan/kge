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
    第三种，直接adae过一遍就好了

    这里需要在embedding赋值的时候 直接复制而不是.detach，什么时候需要detach，而是picker的输入需要，只在哪里detach即可。
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
    2023/09/23
    现在决定了，piceker应该依附于multi_embedder,因为设计之初，就期望AdaE产出的结果是类似于原来模型的embeddding,并且其输入也是index,输出也是embedding
    可以跑通了，明天开始进行双层规划的测试
    2023/09/24
    上面的想法很好，但是导致optimizer_c需要修改，这可咋办,
    发现不需要_entity_embedder
    2023/09/25
    要写penalty这里的loss在architetrue中,并且要在AdaE里写一个process_subbatch一样的_loss出来。
    不能直接用job的process,因为你需要使用传入archt的model，而不是一个不知道哪里来的job中的model
    
    rank_e写的有bug，不同的embedder需要不同的embedder
    倒数模型中输入的p index超过了num_r，因为加了num_r,今天发现是base_model中的dataset被relation*2了，重写relation_reciprocal_model，完美解决。

    optimizer的解决方式是重写一个就ok了。否则architr里在求解picker——params的grad会全部为None，因为无法捕获到params参数
    跑通了
    2023/0928
    服务器跳板机坏了，画了一点点图
    2023/0929
    eval代码重写，发现是adaE()的问题，只要通过if_training判断是否eval，采用original的方式就不会报错。
    查看一下明天的结果，其中最后跑的adae是negative_sampling+original的，最好能与complex持平.
    2023/0930
    不行，发现是AdaE的问题，应该是继承的时候父类初始化函数写错了，检查一下所有新写的集成的类,并且有一处移植spo函数出错了。
    original可以达到以前的 要求了
    2023/1004
    KGE_model_parameters有哪些参数：
    '_base_model._entity_embedder._embeddings.weight'
    <!-- '_base_model._entity_embedder.BN.weight'、bais -->
    '_base_model._entity_embedder.picker.FC1、2.weight'、bais
    '_base_model._entity_embedder.picker.bucket.weight'
    <!-- '_base_model._entity_embedder.Transform_layer_list.0、1.weight'、bais -->
    <!-- '_base_model._entity_embedder._embeddings_list.0、1' -->
    发现picker中计算loss的过程，其中有导致了embeddings的更新。

    2023/1005
    embeddings有几个用处：
    picker的输入 ，已经解决,采用embeddings.detach(),也就是上一次的值
    penalty的输入
    首先输入embeddings的赋值操作无法追踪到embedding_list，因此必须直接使用adaE()方法，但是只进行了部分uniq_index的重写。 现在penalty可以更新到embedding_list了。现在都解决了
    和_loss的求penalty
    因为比较繁琐而且双层规划本身就是一种正则化，并且penalty用处也不是很大、。所以不求了，错最后都求了

    只要设计取embedding，emb()就会赋值 ：有好几处loss的计算属于bi-level的过程，并不需要更改embeddings，赋值只存在于kge_model的aligment的过程。 已经解决
    现在发现picker部分的grad很小-11次方,换成了relu有所缓解
    step让gumbel softmax温度解决了一部分,relation_emb和entity_emb速度为2：1
    2023/1008
    重新调整了架构，rank下加入了ts和zp两种对齐方式，并在zp下设置断言保证不报错
    auto也完成了ts和zp两种对齐方式。也就是说，aligment_way完成了

    


    

    
    

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
  在trainjob中，首先在config-default.yaml中是通过train.type=negative_sampling，找到对应的类
  negative_sampling:
    class_name: TrainingJobNegativeSampling
  因此双层规划的训练方式TrainDarts则通过该方法加入到config-default.yaml中



  ## save模块
  需要吧保存的文件名字后面加上一些变体的名字。
  ## Dataset/训练集的数据分割
  完成了negative_sample里的get_collate函数，处理它
  - 测试模块
  - KgeSampler采样任务

  ## device需要设计
  原代码使用的方法是.to(self.device),此处的self.device指的是（job.device）
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
  ## 0
  服务器显卡掉了
  在user的~下面 sudo bash NVIDIA-...
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
  # 可以优化内存
  把四个embedder中用于存储临时变量的，其中的picker删除掉
  # kvsall是如何与negative_sampling同时初始化的，这一点可以实现darts继承多个类
  # 目前sub_batch和batch一样大，不会出错，但是如果变了，train_darts里architectrue的loss计算要修正
  # 特别隐藏的问题，archt更新的时候用的是batch的全部，而非只有batch,因此一轮中batch_t会被使用两次，第一次是archt,第二次是model
  # train_darts中archi的实例化在init中，而且在not forward_only中，因此在test会出错 
  ## fix解决了
  fix的问题在于，忘记之前写的eval的判断了。
  ## eval的时候，需要再过一遍或者选上一次，这个也顺便解决了eval报错的问题
  现在要选择在eval的时候直接选上一次的好还是，再过一遍好。(再过一遍比较慢，上一次比较快，但也不费多少时间)
  再过一次的实现了，gumbel softmax需要换成一个argmax的方式。，已经实现在mode==auto中。
  ## embeddings设计的问题
  ### picker的输入
  用的上一次embeddings.detach()
  ### penalty计算
  目前使用的是过一遍
  ### _loss计算
  在bi-level的计算中，会有赋值的操作会更改embeddings的值，因此加了一个self.is_bilevel的判断。
  ### architecture.step()的更新正常，KGE部分不更新，但是Picker更新。embeddings都不更新。
  目前picker中FC1的梯度比较小，但是更新正常
  archi的三部分梯度更新正常
  _embedding_list[0]的grad在penalty梯度更新正常
  KGE_model的更新正常,embeddings也正常,三部分正常
  ### 新写了一个cli_debug_auto.py
  唯一的区别是python命令的时候save的不同意见args的不同
  ### 富威师兄的意见
  ConvE试一试
  zero_padding
  fix的不同大小也试一试 get，发现学习率调整大一点会好一点

  ### zero_padding
  出现了[64,1024]到256有问题，因为1024无法zero_padding
  所以对齐的维度要>=dim_list[-1],目前在跑1024




  做eye初始化，load

  convE调试写好代码跑

  zero_padding只适合share_rank吧，毕竟初er始化的方法是uniform,所以先看看share_rank的写法
  share的话，ts在测试会报错不知道为什么
  ### noAF_drop
  noAF_drop会好很多 
  现有任务
  rank【1024,1024】在跑两个 写了
  noshare_zp正在搞 两个
  share_ts报错了
  share_zp回去跑两个 跑了，最好
  conve可以调试一下

  ### rank现在没有吧全部的训练数据用于其中，而是80%，每次训练（但是会打乱）

  conve跑了rank
  跑了auto
  今晚上写完全部用这个的版本，加上三种分类
  dataset.count_e_rank中的save重新写一下智能点，写完了
  发现训练方式选的是1vsall。明天重新跑一遍（把rank改完,需要手动改正）
  zp训练完了，结果记录一下
  把test的库写出来


  zp的结果看一下
  [64,256]的结果看一下
  其他的对齐方法
  看一下复数的影响  ，已经修改完成，看来需要对不同的emb进行不同的dropout，但是很难调试成果
  share和zp必须搭配，初始化可以被share消除掉，因此不需要再进行LN 有两个实验
  如果不share，那么zp必须之后进行LN    有两个实验
  BN如何使用？
  ### 一旦[1024,256],[256,1024]ts不进行BN效果奇差，为什么？
  ### 对齐=1024
  试一试更小的学习率和初始化
  ### 训练方式更换
  直接换继承的顺序，adae_config
  ### adae
  调试发现darts中二阶导数有问题，直接舍弃,后来发现没问题
  其实是学习率的问题，必须适合adagrad，也就是0.01左右
  ### 现在改成了picker双层网络，发现学习有曲线
  ### 新增了重新训练阶段
  必须实验cli_retrain.py
  命令格式如下：
  python kge/cli_retrain.py models/fb15k-237/AdaE_auto.yaml cuda:4 '[128,256,1024]' 0.5 0.5  '20231017-113833-AdaE_auto-auto-noshare-[0.02, 0.04, 0.08, 0.16, 0.32, 0.64]-[128, 256, 1024]-ts-LN-1vsall-unrolled--256-0.5-0.5'

  # 整除写一下
  save改一下
  删掉optimizer_p那一块
  # 停止命令
  pkill -u  guanzp python
  pkill  -u guanzp -f "python kge/cli_debug.py models/fb15k-237/AdaE_rank.yaml cuda:4"
  
# 随机选择
## trick
 tau=0.001，pretrain,+lr=0.000053，可以好好选 ,
 

 最重要的是温度

 tau=1/ 0.1，pretrain,+lr=0.000053/0.053，soft=false,最重要，可以选
 tau=1~ 0.1，+lr=0.053，soft=false,最重要，可以选
 tau越小越快，越容易达到one-hot

 ## 加了噪声 
 lr=0.53,share
 picker输入128+emb，两个picker，无penalty_m
 很快选择到最好的size，但是同质化严重
 ## twins
 网络解决这个问题，发现 ，能一定程度解决前期立刻同志到一个，但是到了中期则会集中到同一个
 ## 加了随机选择概率，退火变小 ，
 随机概率fixed=0.5的时候，有一定几率可以选两个了。
 去掉这个试一试下面的

 ## shared 的问题
 share，还是选到96上了
 ## dropout
 0.5还是不行
 ## 临近的size，看一看是否选的都是最大的？
 [76,80]

 ## 固定住某些部分
  ### relation_embedder
  没有
  ### bucket_emb
  没有,会让模型收敛得更慢
  

  ## 加了输入
  padding-0，a
  rate=0.01~0.2，会选择两个size，跑出来，每一个网络选了一个出来.
  rate=0.5, 还是选出来一个size。 
  padding-a
  rate=0.001, 
  早停准则