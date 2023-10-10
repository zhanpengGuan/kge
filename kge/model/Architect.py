
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from copy import deepcopy

def _concat(xs):
  return torch.cat([x.view(-1) for x in xs]) #把x先拉成一行，然后把所有的x摞起来，变成n行


class Architect(object):

  def __init__(self, model, picker_params, optimizer_c, job, args):
    self.network_momentum = 0
    self.args = args
    self.model = model
    # self.arch_parameters(self.model) = picker_params
    self.device = self.args.get("job.device")
    self.optimizer = optimizer_c
    self.job = job
    """
    我们更新梯度就是theta = theta + v + weight_decay * theta 
      1.theta就是我们要更新的参数
      2.weight_decay*theta为正则化项用来防止过拟合
      3.v的值我们分带momentum和不带momentum：
        普通的梯度下降：v = -dtheta * lr 其中lr是学习率，dx是目标函数对x的一阶导数
        带momentum的梯度下降：v = lr*(-dtheta + v * momentum)
    """
    #【完全复制外面的Network更新w的过程】，对应公式6第一项的w − ξ*dwLtrain(w, α)
    #不直接用外面的optimizer来进行w的更新，而是自己新建一个unrolled_model展开，主要是因为我们这里的更新不能对Network的w进行更新
  def _compute_unrolled_model(self, batch_index, batch_t, eta, network_optimizer, mode='single'):
    
    loss = self.model._loss(batch_index, batch_t) #Ltrain
    theta = _concat(self.KGE_parameters(self.model)).data #把参数整理成一行代表一个参数的形式,得到我们要更新的参数theta/w
    try:
      moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.KGE_parameters(self.model)).mul_(self.network_momentum) #momentum*v,用的就是Network进行w更新的momentum
    except:
      moment = torch.zeros_like(theta) #不加momentum

    dtheta = _concat(torch.autograd.grad(loss.avg_loss, self.KGE_parameters(self.model), allow_unused=True)).data  #前面的是loss对参数theta/w求梯度，self.network_weight_decay*theta就是正则项
    #对参数进行更新，等价于optimizer.step()
    unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment+dtheta))  
    #w − ξ*dwLtrain(w, α) 
                                                                                      
    return unrolled_model



  def step(self, batch_index, batch_t, batch_v, eta, network_optimizer,  unrolled=True, mode='single'):
    self.model._entity_embedder.is_bilevel = True
    self.model._relation_embedder.is_bilevel = True
    
    # input_train, target_train, input_valid, target_valid = positive_sample_train, negative_sample_train, positive_sample_search, negative_sample_search
    #network_opt 指的是w的optmizer

    self.optimizer.zero_grad()#清除上一步的残余更新参数值
    if  unrolled:#用论文的提出的方法
        self._backward_step_unrolled(batch_index, batch_t, batch_v, eta, network_optimizer,mode)
    else: #不用论文提出的bilevel optimization，只是简单的对α求导
        self._backward_step(batch_index, batch_v)
    
    self.optimizer.step() #应用梯度：根据反向传播得到的梯度进行参数的更新， 这些parameters的梯度是由loss.backward()得到的，optimizer存了这些parameters的指针
                          #因为这个optimizer是针对alpha的优化器，所以他存的都是alpha的参数
    self.model._entity_embedder.is_bilevel = False
    self.model._relation_embedder.is_bilevel = False
    return self.optimizer

  def _backward_step(self, batch_index, batch_v):
    
    loss = self.model._loss(batch_index, batch_v)
    loss.avg_loss.backward()

  
  def _backward_step_unrolled(self, batch_index, batch_t, batch_v, eta, network_optimizer,mode):
    if self.args['type']=='1vsall':
      input_train, target_train, input_valid, target_valid, = batch_t["triples"], None, batch_v['triples'], None
    elif self.args['type']=='ng_sampling':
      input_train, target_train, input_valid, target_valid, = batch_t["triples"], batch_t["negative_samples"], batch_v['triples'], batch_v["negative_samples"]
    #计算公式六：dαLval(w',α) ，其中w' = w − ξ*dwLtrain(w, α)
    #w'
    unrolled_model = self._compute_unrolled_model(batch_index, batch_t, eta, network_optimizer, mode)#unrolled_model里的w已经是做了一次更新后的w，也就是得到了w'
    #Lval
    unrolled_loss = unrolled_model._loss(batch_index, batch_v) #对做了一次更新后的w的unrolled_model求验证集的损失，Lval，以用来对α进行更新
    ### 这里之前对了
    unrolled_loss.avg_loss.backward()
    # dαLval(w',α)
        
    self.arch_params = self.arch_parameters(self.model)
    # dαLval(w',α)
    dalpha = [v.grad for v in self.arch_parameters(unrolled_model)] #对alpha求梯度
    # dw'Lval(w',α)
    vector = [v.grad.data for v in self.KGE_parameters(unrolled_model)] #unrolled_model.parameters()
    #共16个参数，entity_embedder8个，relation八个，得到w‘
    #计算公式八(dαLtrain(w+,α)-dαLtrain(w-,α))/(2*epsilon)   其中w+=w+dw'Lval(w',α)*epsilon w- = w-dw'Lval(w',α)*epsilon
    implicit_grads = self._hessian_vector_product(vector, batch_index, batch_t)

    # 公式六减公式八 dαLval(w',α)-(dαLtrain(w+,α)-dαLtrain(w-,α))/(2*epsilon)
    for g, ig in zip(dalpha, implicit_grads):
      g.data.sub_(eta, ig.data)
    #对α进行更新
    for v, g in zip(self.arch_parameters(self.model), dalpha):
      if v.grad is None:
        v.grad = Variable(g.data)
      else:
        v.grad.data.copy_(g.data)
  #对应optimizer.step()，对新建的模型的参数进行更新
  def _construct_model_from_theta(self, theta):
    # theta/w = w − ξ*dwLtrain(w, α) 
    model_new = deepcopy(self.model)
    model_dict = model_new.state_dict() #Returns a dictionary containing a whole state of the module.

    params, offset = {}, 0
    for k, v in self.nKGE_parameters(self.model):#k是参数的名字，v是参数
      v_length = np.prod(v.size())
      params[k] = theta[offset: offset+v_length].view(v.size()) #将参数k的值更新为theta对应的值
      offset += v_length

    assert offset == len(theta)
    model_dict.update(params) #模型中的参数已经更新为做一次反向传播后的值
    model_new.load_state_dict(model_dict,strict=False) #恢复模型中的参数，也就是我新建的mode_new中的参数为model_dict
    return model_new.to(self.device)

  
  #计算公式八(dαLtrain(w+,α)-dαLtrain(w-,α))/(2*epsilon)   其中w+=w+dw'Lval(w',α)*epsilon w- = w-dw'Lval(w',α)*epsilon
  def _hessian_vector_product(self, vector, batch_index, batch_t, r=1e-2): # vector就是dw'Lval(w',α)
    R = r / _concat(vector).norm() #epsilon

    #dαLtrain(w+,α)
    for p, v in zip(self.KGE_parameters(self.model), vector):
      p.data.add_(R, v) #将模型中所有的w'更新成w+=w+dw'Lval(w',α)*epsilon
    loss = self.model._loss(batch_index, batch_t)
    grads_p = torch.autograd.grad(loss.avg_loss, self.arch_parameters(self.model), allow_unused=True)

    #dαLtrain(w-,α)
    for p, v in zip(self.KGE_parameters(self.model), vector):
      p.data.sub_(2*R, v) #将模型中所有的w'更新成w- = w+ - (w-)*2*epsilon = w+dw'Lval(w',α)*epsilon - 2*epsilon*dw'Lval(w',α)=w-dw'Lval(w',α)*epsilon
    loss = self.model._loss(batch_index, batch_t)
    grads_n = torch.autograd.grad(loss.avg_loss, self.arch_parameters(self.model), allow_unused=True)

    #将模型的参数从w-恢复成w
    for p, v in zip(self.KGE_parameters(self.model), vector):
      p.data.add_(R, v) #w=(w-) +dw'Lval(w',α)*epsilon = w-dw'Lval(w',α)*epsilon + dw'Lval(w',α)*epsilon = w

    return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]
  
  def arch_parameters(self,model,recurse: bool = True):
    '''
    a parameter filter for architecture parameters
    '''
    for name, param in model.named_parameters(recurse=recurse):
      if name.startswith('_base_model._relation_embedder.picker') or name.startswith('_base_model._entity_embedder.picker') :
        yield param
  def KGE_parameters(self,model,recurse: bool = True):
    '''
    a parameter filter for KGE parmeters
    '''
    for name, param in model.named_parameters(recurse=recurse):
      if  not (name.startswith('_base_model._entity_embedder._embeddings.weight') or name.startswith('_base_model._relation_embedder._embeddings.weight') or name.startswith('_base_model._relation_embedder.picker') or name.startswith('_base_model._entity_embedder.picker')) :
        yield param
  def nKGE_parameters(self,model,recurse: bool = True):
    '''
    a named parameter filter for KGE parmeters
    '''
    for name, param in model.named_parameters(recurse=recurse):
      if  not (name.startswith('_base_model._entity_embedder._embeddings.weight') or name.startswith('_base_model._relation_embedder._embeddings.weight') or name.startswith('_base_model._relation_embedder.picker') or name.startswith('_base_model._entity_embedder.picker')) :
        yield name, param