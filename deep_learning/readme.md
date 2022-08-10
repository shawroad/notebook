- [1. pytorch保存并加载checkpoint](#1-pytorch保存并加载checkpoint)
- [2. focal_loss用于多标签分类、多分类。](#2-focal_loss用于多标签分类多分类)
- [3. pytorch对向量的基本操作(复制、变换维度)](#3-pytorch对向量的基本操作复制变换维度)
- [4. Pytorch模型的训练可视化](#4-pytorch模型的训练可视化)
- [5. 取出预训练模型的中间层](#5-取出预训练模型的中间层)
- [6. 加快pytorch中数据的读取过程](#6-加快pytorch中数据的读取过程)
- [7. 将pytorch模型转为onnx加快推理](#7-将pytorch模型转为onnx加快推理)
  - [第一步: 先进行转化,并核对和原始模型的输出是否一致](#第一步-先进行转化并核对和原始模型的输出是否一致)
  - [第二步: 加载上面保存的onnx_bert_cls.onnx文件并预测](#第二步-加载上面保存的onnx_bert_clsonnx文件并预测)
- [8. 将pytorch模型转为c++模型进行推理加速](#8-将pytorch模型转为c模型进行推理加速)
  - [第一步: 转为C++](#第一步-转为c)
  - [第二步: 加载model_jit.pt文件](#第二步-加载model_jitpt文件)
- [9. pytorch模型结构的可视化](#9-pytorch模型结构的可视化)
- [10. Transformers中BertTokenizer和BertTokenizerFast的速度对比](#10-transformers中berttokenizer和berttokenizerfast的速度对比)
- [11. DiceLoss](#11-diceloss)
- [12. EMA指数平均](#12-ema指数平均)
- [13. 生成模型的几种解码方式](#13-生成模型的几种解码方式)
- [14. mseloss中引入相关性损失](#14-mseloss中引入相关性损失)
- [15. pytorch中model.train和model.eval以及torch.no_grad()的含义](#15-pytorch中modeltrain和modeleval以及torchno_grad的含义)
- [16. pytorch中张量之间的运算](#16-pytorch中张量之间的运算)
- [17. labelsmooth的实现](#17-labelsmooth的实现)
- [18. 预训练模型微调前加点噪声NoisyTune](#18-预训练模型微调前加点噪声noisytune)
- [19. 模型压缩](#19-模型压缩)
  - [权重压缩](#权重压缩)
  - [权重裁剪](#权重裁剪)

# 1. pytorch保存并加载checkpoint

  ```python
  import torch

  net = ResNet()
  loss = MSE()
  optimizer = torch.optim.SGD(net.parameters(), lr=0.001)
  """
  正常训练的代码
  """
  # net为某个模型
  net.train()
  for epoch in range(total_epochs):
     ...
     for x, y in dataloader:
         ...
         loss = ...
         loss.backward()
         optimizer.step()
     state_dict = {"net": net.state_dict(),
                   "optimizer": optimizer.state_dict(),
                   "epoch": epoch}
     torch.save(state_dict, "model_path/model.pth")

  """
  中断后，加载权重继续训练
  """
  checkpoint = torch.load("model_path/model.pth")
  current_epoch = checkpoint["epoch"]
  net.load_state_dict(checkpoint['net'])
  optimizer.load_state_dict(checkpoint['optimizer'])
  net.train()
  for epoch in range(current_epoch , total_epochs):
     ...
     for x, y in dataloader:
         ...
         loss = ...
         loss.backward()
         optimizer.step()
     state_dict = {"net": net.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}
     torch.save(state_dict, "model_path/model.pth")
  ```

# 2. focal_loss用于多标签分类、多分类。

  ```python
  import torch
  from torch import nn
  import torch.nn.functional as F


  class BCEFocalLoss(nn.Module):
      # 可用于二分类和多标签分类
      def __init__(self, gamma=2, alpha=0.25, reduction='mean'):
          super(BCEFocalLoss, self).__init__()
          self.gamma = gamma
          self.alpha = alpha
          self.reduction = reduction

      def forward(self, logits, labels):
          '''
          假设是三个标签的多分类
          loss_fct = BCEFocalLoss()
          labels = torch.tensor([[0, 1, 1], [1, 0, 1]])
          logits = torch.tensor([[0.3992, 0.2232, 0.6435],[0.3800, 0.3044, 0.3241]])
          loss = loss_fct(logits, labels)
          print(loss)  # tensor(0.0908)

          '''
          probs = torch.sigmoid(logits)

          loss = -self.alpha * (1 - probs) ** self.gamma * labels * torch.log(probs) - (1 - self.alpha) * probs ** self.gamma * (1 - labels) * torch.log(1 - probs)

          if self.reduction == 'mean':
              loss = torch.mean(loss)
          elif self.reduction == 'sum':
              loss = torch.sum(loss)
          return loss


  class MultiCEFocalLoss(nn.Module):
      # 可以用于多分类 (注: 不是多标签分类)
      def __init__(self, class_num, gamma=2, alpha=None, reduction='mean'):
          super(MultiCEFocalLoss, self).__init__()
          if alpha is None:
              self.alpha = torch.ones(class_num, 1)
          else:
              self.alpha = alpha
          self.gamma = gamma
          self.reduction = reduction
          self.class_num = class_num

      def forward(self, logits, labels):
          '''
          logits: (batch_size, class_num)
          labels: (batch_size,)
          '''
          probs = F.softmax(logits, dim=1) 
          class_mask = F.one_hot(labels, self.class_num)   # 将真实标签转为one-hot
          ids = labels.view(-1, 1)   # (batch_size, 1)
          alpha = self.alpha[ids.data.view(-1)]   # 每一类的权重因子

          probs = (probs * class_mask).sum(1).view(-1, 1)
          log_p = probs.log()

          loss = -alpha * (torch.pow((1-probs), self.gamma)) * log_p

          if self.reduction == 'mean':
              loss = loss.mean()
          elif self.reduction == 'sum':
              loss = loss.sum()

          return loss

  if __name__ == '__main__':
      # loss_fct = BCEFocalLoss()
      # labels = torch.tensor([[0, 1, 1], [1, 0, 1]])
      # logits = torch.tensor([[0.3992, 0.2232, 0.6435],[0.3800, 0.3044, 0.3241]])
      # loss = loss_fct(logits, labels)
      # print(loss)

      # 举例四分类
      loss_fct = MultiCEFocalLoss(class_num=4)
      labels = torch.tensor([1, 3, 0, 0, 2])
      logits = torch.randn(5, 4)
      loss = loss_fct(logits, labels)
      print(loss)
  ```

# 3. pytorch对向量的基本操作(复制、变换维度)

  - 将向量沿着某些维度展开，实现一下simcse中inbatch复制的方式,即: [[1, 2, 3], [4, 5, 6]] 搞成 [[1, 2, 3],[1, 2, 3],[4, 5, 6],[4, 5, 6]]

  ```python
  import torch

  if __name__ == '__main__':
      data = torch.tensor([[1, 2, 3], [4, 5, 6]])   # size: (batch_size, max_len)
      data = data.unsqueeze(1)   # size: (batch_size, 1, max_len) 要对max_len那一维进行横向复制，则在前面加一维

      # 接下来使用repeat进行复制  如果你对几维的张量进行复制，则对每个维度都要指定复制多少遍
      # 从下面的例子中可以看到，对batch_size复制一遍(复制一遍相当不动), 对第二维复制两边，相当于叠加一次
      res = data.repeat(1, 2, 1)   # size: (batch_size, 2, max_len)

      # 将多复制的那一边叠加到batch中
      res = res.view(-1, data.size(-1))   # size: (batch_size * 2, max_len)
      print(res)
      """
      输出: 
      tensor([[1, 2, 3],
              [1, 2, 3],
              [4, 5, 6],
              [4, 5, 6]])
      """
  ```

  - 再来一个例子，将[[1, 2, 3], [4, 5, 6]]搞成[[1, 1, 2, 2, 3, 3], [4, 4, 5, 5, 6, 6]]

  ```python
  import torch

  if __name__ == '__main__':
      data = torch.tensor([[1, 2, 3], [4, 5, 6]]) 
      # size: (batch_size, max_len)
      # 难道直接用下面那行代码 显然不对 
      print(data.repeat(1, 2))
      '''
      输出:
      tensor([[1, 2, 3, 1, 2, 3],
              [4, 5, 6, 4, 5, 6]])
      '''

      data = data.unsqueeze(-1)
      print(data.size())   # torch.Size([2, 3, 1])
      print(data)
      '''
      输出:
      tensor([[[1], [2], [3]], [[4], [5], [6]]])
      '''
      data = data.repeat(1, 1, 2)
      print(data.size())   # torch.Size([2, 3, 2])
      print(data)
      '''
      输出:
      tensor([[[1, 1], [2, 2], [3, 3]], [[4, 4], [5, 5], [6, 6]]])
      '''
      data = data.view(data.size(0), -1) 
      print(data)
      '''
      tensor([[1, 1, 2, 2, 3, 3],
              [4, 4, 5, 5, 6, 6]])
      '''
  ```

# 4. Pytorch模型的训练可视化

  在pytorch模型进行训练的过程中，需要实时观测损失、准确率等的变化。直接借助tensorboard可观看。

  ```python
  # 模型部分
  from torch import nn
  class ConvNet(nn.Module):
      def __init__(self):
          super(ConvNet, self).__init__()
          self.conv1 = nn.Sequential(
              nn.Conv2d(1, 16, 3, 1, 1),
              nn.ReLU(),
              nn.AvgPool2d(2, 2)
          )
   
          self.conv2 = nn.Sequential(
              nn.Conv2d(16, 32, 3, 1, 1),
              nn.ReLU(),
              nn.MaxPool2d(2, 2)
          )
  
          self.fc = nn.Sequential(
              nn.Linear(32 * 7 * 7, 128),
              nn.ReLU(),
              nn.Linear(128, 64),
              nn.ReLU()
          )

          self.out = nn.Linear(64, 10)
  
      def forward(self, x):
          x = self.conv1(x)
          x = self.conv2(x)
          x = x.view(x.size(0), -1)
          x = self.fc(x)
          output = self.out(x)
          return output
    
  # 训练过程
  import torch
  import torchvision
  from torch import nn
  from model import ConvNet
  import torch.utils.data as Data
  from sklearn.metrics import accuracy_score
  from tensorboardX import SummaryWriter

  '''
  安装:
  pip install tensorboardX
  pip install tensorboard

  代码运行后，在和run在同一个目录下的命令行上输入:
  tensorboard --logdir="./data/log"
  即可看到准确率和损失的变化
  '''

  if __name__ == '__main__':
      MyConvNet = ConvNet()

      # 准备训练用的MNIST数据集
      train_data =   torchvision.datasets.MNIST(
          root="./data/MNIST",  # 提取数据的路径
          train=True,  # 使用MNIST内的训练数据
          transform=torchvision.transforms.ToTensor(),  # 转换成torch.tensor
          download=False  # 如果是第一次运行的话，置为True，表示下载数据集到root目录
      )

      # 定义loader
      train_loader = Data.DataLoader(
          dataset=train_data,
          batch_size=64,
          shuffle=True,
          num_workers=0
      )

      test_data = torchvision.datasets.MNIST(
          root="./data/MNIST",
          train=False,  # 使用测试数据
          download=False
      )

      # 将测试数据压缩到0-1
      test_data_x = test_data.data.type(torch.FloatTensor) / 255.0
      test_data_x = torch.unsqueeze(test_data_x, dim=1)
      test_data_y = test_data.targets

      # 打印一下测试数据和训练数据的shape
      print("test_data_x.shape:", test_data_x.shape)   # torch.Size([10000, 1, 28, 28])
      print("test_data_y.shape:", test_data_y.shape)   # torch.Size([10000])

      logger = SummaryWriter(log_dir="data/log")

      # 获取优化器和损失函数
      optimizer = torch.optim.Adam(MyConvNet.parameters(), lr=3e-4)
      loss_func = nn.CrossEntropyLoss()
      log_step_interval = 100  # 记录的步数间隔

      for epoch in range(5):
          print("epoch:", epoch)
          # 每一轮都遍历一遍数据加载器
          for step, (x, y) in enumerate(train_loader):
              # 前向计算->计算损失函数->(从损失函数)反向传播->更新网络
              predict = MyConvNet(x)
              loss = loss_func(predict, y)
              optimizer.zero_grad()  # 清空梯度（可以不写）
              loss.backward()  # 反向传播计算梯度
              optimizer.step()  # 更新网络
              global_iter_num = epoch * len(train_loader) + step + 1  # 计算当前是从训练开始时的第几步(全局迭代次数)
              if global_iter_num % log_step_interval == 0:
                  # 控制台输出一下
                  print("global_step:{}, loss:{:.2}".format(global_iter_num, loss.item()))
                  # 添加的第一条日志：损失函数-全局迭代次数
                  logger.add_scalar("train loss", loss.item(), global_step=global_iter_num)
                  # 在测试集上预测并计算正确率
                  test_predict = MyConvNet(test_data_x)
                  _, predict_idx = torch.max(test_predict, 1)  # 计算softmax后的最大值的索引，即预测结果
                  acc = accuracy_score(test_data_y, predict_idx)
                  # 添加第二条日志：正确率-全局迭代次数
                  logger.add_scalar("test accuary", acc.item(), global_step=global_iter_num)
                  # 添加第三条日志：网络中的参数分布直方图
                  for name, param in MyConvNet.named_parameters():
                      logger.add_histogram(name, param.data.numpy(), global_step=global_iter_num)
  ```

# 5. 取出预训练模型的中间层

demo: 取出resnet的中间层特征

```python
import torch
import torchvision
from torch import nn


class FeatureExtractor(nn.Module):
    # 中间层特征提取
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs = []
        for name, module in self.submodule._modules.items():
            if name is "fc":
                x = x.view(x.size(0), -1)

            x = module(x)
            if name in self.extracted_layers:
                outputs.append(x)
        return outputs


if __name__ == '__main__':
    # 1. 定义要抽取的层
    extracted_layers = ['layer3', 'layer4', 'avgpool']

    # 2. resnet模型
    resnet = torchvision.models.resnet50(pretrained=True)

    # 3. 随机产生数据
    data = torch.randint(255, size=(2, 3, 225, 225)).float()

    # 4. 定义特征抽取的模型
    model = FeatureExtractor(resnet, extracted_layers)

    res = model(data)
    for i in res:
        print(i.size())
```

另外一种写法  可以自己定义加载预训练   做分类

```python
import torch
from torch import nn
from torchvision.models import resnet50


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.submodule = resnet50(pretrained=False)
        self.load_pretrained(self.submodule)
        self.extracted_layers = "avgpool"
        self.classify = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        batch_size = x.size()[0]
        for name, module in self.submodule._modules.items():
            if name is "fc":
                x = x.view(x.size(0), -1)

            x = module(x)
            if name == self.extracted_layers:
                x = x.view(batch_size, -1)
                output = self.classify(x)
                return output

    def load_pretrained(self, sub_model):
        sub_model.load_state_dict(torch.load('./resnet_pretrain/resnet50-0676ba61.pth'))
        return sub_model


if __name__ == '__main__':
    torch.manual_seed(43)
    input_data = torch.randn(size=(1, 3, 224, 224))
    print(input_data.size())
    model = Model()
    res = model(input_data)
    print(res.size())
```

# 6. 加快pytorch中数据的读取过程

通常，在使用pytorch写模型的时候，使用DataSet类、DataLoader类来加载数据。 这一段一般运行在cpu上，模型训练跑在gpu上，有可能cpu会影响数据的加载，最终导致模型训练变慢。可以通过下面的两个策略加快数据的读取，是cpu利用率得到提高。

1. 在DataLoader类中加入:`pin_memory=True`，锁页内存，内存够用一定要加上。科普一下：主机中的内存，有两种存在方式，一是锁页，二是不锁页，锁页内存存放的内容在任何情况下都不会与主机的虚拟内存进行交换（注：虚拟内存就是硬盘），而不锁页内存在主机内存不足时，数据会存放在虚拟内存中。而显卡中的显存全部是锁页内存，设置为True时可以加快内存和显卡之前的数据交换；
2. 在DataLoader类中加入: `prefetch_factor=任意数字，默认=2。` 代表的含义是每个 worker 提前加载 的 sample 数量，可以适当调大，具体改调到多大合适还没做过实验，我觉得根据batch size=prefetch_factor*num_workers比较合理
3. persistent_workers=True，看官方文档的意思就是默认Flase相当于所有数据都训练完这一轮之后（跑完一次epoch）worker会自动关闭，下个epoch需要重新打开并初始化worker，这会浪费一点时间。该参数可以控制每个epoch的第一个iteration（第一批batch）是否重新初始化worker，所以可以通过设置=True去除每个epoch第一个iteration多消耗的一点时间；

num_wokers别忘了加入。num_woker是用来指定用几个进程加载数据。

```python
# 训练数据集准备
train_dataset = CustomDataset(dataframe=train_df, tokenizer=tokenizer)
train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=args.train_batch_size,
                              collate_fn=collate_fn, pin_memory=True, num_workers=3)
```

# 7. 将pytorch模型转为onnx加快推理

## 第一步: 先进行转化,并核对和原始模型的输出是否一致

```python
import torch
from model import Model
import tempfile
import onnx
import numpy as np
from onnxruntime import InferenceSession
import time
# pip install onnx==1.11.0 onnxruntime==1.10.0


def export_to_onnx():
    # 下面的dummy_input样本个数可以是一条 也可以是多条
    dummy_input = {
        "input_ids": torch.tensor([[101, 231, 353, 123, 642, 5452, 123, 102],
                                   [101, 231, 353, 123, 642, 5452, 123, 102],], dtype=torch.long),
        "attention_mask": torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.long),
        "segment_ids": torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.long)
    }

    dynamic_axes = {
        'input_ids': [0, 1],
        'attention_mask': [0, 1],
        'token_type_ids': [0, 1],
    }
    output_names = ["logits"]

    with tempfile.NamedTemporaryFile() as fp:
        torch.onnx.export(model,
                          args=tuple(dummy_input.values()),
                          f=fp,
                          input_names=list(dummy_input),
                          output_names=output_names,
                          dynamic_axes=dynamic_axes,
                          opset_version=11)
        sess = InferenceSession(fp.name)
        model.eval()

        start_time = time.time()
        logits_ori = model(**dummy_input.copy())

        mid_time = time.time()
        logits_onnx = sess.run(
            output_names=output_names,
            input_feed={key: value.numpy() for key, value in dummy_input.items()})
        end_time = time.time()

        np.testing.assert_almost_equal(logits_ori.detach().numpy(), logits_onnx[0], 5)
        print('原始pytorch花费时间(GPU): ', mid_time - start_time)
        print('转为onnx花费时间(GPU):', end_time - mid_time)

    # 把onnx模型保存下来
    torch.onnx.export(model,
                      args=tuple(dummy_input.values()),
                      f='onnx_bert_cls.onnx',
                      input_names=list(dummy_input),
                      output_names=output_names,
                      dynamic_axes=dynamic_axes,
                      opset_version=11)


if __name__ == '__main__':
    model = Model(15)
    model.load_state_dict(torch.load('./outputs/base_model_epoch_0.bin'))   # 加载训练好的权重
    model.eval()
    export_to_onnx()
```

时间上的对比:  
原始pytorch花费时间(CPU):  0.1219179630279541  
转为onnx花费时间(CPU): 0.05363821983337402  

## 第二步: 加载上面保存的onnx_bert_cls.onnx文件并预测

```python
import torch
import onnx
from onnxruntime import InferenceSession


# 假设保存完毕 加载onnx
onnx_model = onnx.load("onnx_bert_cls.onnx")
sess = InferenceSession("onnx_bert_cls.onnx")
dummy_input = {
    "input_ids": torch.tensor([[101, 231, 353, 123, 642, 5452, 123, 102],
                               [101, 231, 353, 123, 642, 5452, 123, 102]], dtype=torch.long),
    "attention_mask": torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.long),
    "segment_ids": torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.long)
}
output_names = ["logits"]
logits_onnx = sess.run(
    output_names=output_names,
    input_feed={key: value.numpy() for key, value in dummy_input.items()})
print(logits_onnx)
```

最后可以可视化该模型

```
import netron
netron.start('lenet.onnx')
```

# 8. 将pytorch模型转为c++模型进行推理加速

## 第一步: 转为C++

```python
import torch
from model import Model


def export_jit_model():
    # "segment_ids"
    input_ids = torch.tensor([[101, 231, 353, 123, 642, 5452, 123, 102]], dtype=torch.long)
    attention_mask = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.long)
    segment_ids = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.long)

    jit_sample = [input_ids, attention_mask, segment_ids]

    module = torch.jit.trace(model, jit_sample)

    # 保存一下模型
    module.save('./model_jit.pt')

    print('C++推理:')
    logits_c = module(input_ids, attention_mask, segment_ids)
    print(logits_c)
    print('原始Pytorch推理:')
    logits_p = model(input_ids, attention_mask, segment_ids)
    print(logits_p)


if __name__ == '__main__':
    model = Model(15)
    model.load_state_dict(torch.load('./outputs/base_model_epoch_0.bin'))
    model.eval()
    export_jit_model()
```

## 第二步: 加载model_jit.pt文件

```python
import torch


if __name__ == '__main__':
    input_ids = torch.tensor([[101, 231, 353, 123, 642, 5452, 123, 102]], dtype=torch.long)
    attention_mask = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.long)
    segment_ids = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.long)

    model = torch.jit.load('./model_jit.pt')
    logits = model(input_ids, attention_mask, segment_ids)
    print(logits)
```

# 9. pytorch模型结构的可视化

`pip install torchinfo ` 或者 `conda install -c conda-forge torchinfo`

```python
from torchvision import models
from torchinfo import summary


if __name__ == '__main__':
    model = models.resnet18()
    print('打印出更详细的模型结构:')
    print(summary(model, (1, 3, 224, 224)))
```

# 10. Transformers中BertTokenizer和BertTokenizerFast的速度对比

```python
import time
from transformers.models.bert import BertTokenizer
from transformers.models.bert import BertTokenizerFast


def test_single_speed():
    sentence1 = '大婚前夕还被未婚夫休弃 再次轮为笑柄 而她 是医术过人的杀手 21世纪的新新人类 被自己深爱的男人枪杀而亡 点击此处继续阅读'
    sentence2 = '大婚前夕还被未婚夫休弃 再次轮为笑柄 而她 是医术过人的杀手 21世纪的新新人类 被自己深爱的男人枪杀而亡 点击此处继续阅读'

    tokenizer_ori = BertTokenizer.from_pretrained('./roberta_pretrain')
    tokenizer_fast = BertTokenizerFast.from_pretrained('./roberta_pretrain')

    s_time = time.time()
    for i in range(10000):
        res = tokenizer_ori(sentence1, sentence2)
    e_time = time.time()
    print(e_time - s_time)    # 花费时间: 7.550800085067749s

    s_time = time.time()
    for i in range(10000):
        res = tokenizer_fast(sentence1, sentence2)
    e_time = time.time()
    print(e_time - s_time)    # 花费时间: 3.5354318618774414s


def test_batch_speed():
    sent1 = ['大鱼大肉吃腻', '大鱼大肉吃腻吗', '大鱼大肉吃腻哈哈哈', '大鱼大肉吃腻鸡毛', '大鱼大肉吃腻', '大鱼大肉吃腻吗', '大鱼大肉吃腻哈哈哈', '大鱼大肉吃腻鸡毛']
    sent2 = ['当代大学生回家', '当代大学生回家', '当代大学生回家呼呼呼呼呼呼', '当代大学生回家个', '当代大学生回家', '当代大学生回家', '当代大学生回家呼呼呼呼呼呼', '当代大学生回家个']

    tokenizer_ori = BertTokenizer.from_pretrained('./roberta_pretrain')
    tokenizer_fast = BertTokenizerFast.from_pretrained('./roberta_pretrain')

    # 批量编码  并进行padding
    s_time = time.time()
    for i in range(10000):
        res = tokenizer_ori(sent1, sent2, max_length=54, truncation=True, padding='max_length')
    e_time = time.time()
    print(e_time - s_time)    # 花费时间: 14.62081789970398s

    s_time = time.time()
    for i in range(10000):
        res = tokenizer_fast(sent1, sent2, max_length=54, truncation=True, padding='max_length')
    e_time = time.time()
    print(e_time - s_time)   # 花费时间: 4.33729100227356


if __name__ == '__main__':
    test_single_speed()   # 每次传入一个
    test_batch_speed()   # 每次传入多个
```

# 11. DiceLoss
$$DiceLoss = 1 - \frac{2|X\cap Y|}{|X| + |Y|}$$

|X|和|Y|分别代表X的所有元素之和和Y的所有元素之和。X∩Y代表的是两个矩阵原始对应位置相乘，然后求和。  后面那一大块描述的就是dice系数。如果两个矩阵越相似，则值越大。 1-dice系数则越小，即:loss。我们的要求肯定是让预测无限接近label.

```python
# ref: https://zhuanlan.zhihu.com/p/68748778
import torch
from torch import nn


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        # 上面计算的dice为dice系数  1-dice系数  就是diceloss
        return 1 - dice


if __name__ == '__main__':
    target = torch.tensor([[0], [1], [2], [1], [0]])   # 三分类

    # 这里必须将标签转为one-hot形式。
    batch_size, label_num = 5, 3
    target_onthot = torch.zeros(batch_size, label_num).scatter_(1, target, 1)

    logits = torch.rand(5, 3)
    loss_func = DiceLoss()
    loss = loss_func(logits, target_onthot)
    print(loss)

```
# 12. EMA指数平均
```python
class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

# 初始化
ema = EMA(model, 0.999)
ema.register()

# 训练过程中，更新完参数后，同步update shadow weights
def train():
    optimizer.step()
    ema.update()

# eval前，apply shadow weights；eval之后，恢复原来模型的参数
def evaluate():
    ema.apply_shadow()
    # evaluate
    ema.restore()

# 这里强调一下  在保存模型的之前 需要ema.apply_shadow()一下，即把ema后的权重更新到模型上，然后再保存。
# 另外: 模型权重的指数滑动平均, 不参加梯度更新，只是记录滑动平均的参数，给预测使用，注意区别于类似adam一类的自适应学习率优化器, 针对一阶二阶梯度的指数滑动平均, 两者完全不同
```


# 13. 生成模型的几种解码方式
```python
import torch
import os
import re
from config import set_args
from transformers.models.gpt2 import GPT2LMHeadModel
from transformers import BertTokenizer

# transformers==4.10.2


def my_greedy_decode(ad):
    # 自己手动实现
    ad = "最美不过下雨天"
    input_ids = tokenizer(ad, return_tensors="pt")["input_ids"]
    # print(input_ids)   # tensor([[ 101, 3297, 5401,  679, 6814,  678, 7433, 1921,  102]])
    # print(input_ids.size())   # torch.Size([1, 9])
    iterations = []
    n_steps = 8  # 进行8步解码
    choices_per_step = 5  # 每一步候选数量

    with torch.no_grad():
        for i in range(n_steps):
            iteration = dict()
            iteration['Input'] = tokenizer.decode(input_ids[0])
            output = model(input_ids=input_ids)
            # print(output.logits.size())  # torch.Size([1, 9, 21128])

            # 取最后一个token的输出
            next_token_logits = output.logits[0, -1, :]

            # print(next_token_logis.size())   # torch.Size([1, 21128])
            next_token_probs = torch.softmax(next_token_logits, dim=-1)
            sorted_ids = torch.argsort(next_token_probs, dim=-1, descending=True)
            # print(sorted_ids)   # tensor([ 5682,   741,   691,  ..., 12518, 11888, 10980])

            for choice_idx in range(choices_per_step):
                # 选取概率最大的五个
                token_id = sorted_ids[choice_idx]
                token_prob = next_token_probs[token_id].cpu().numpy()
                token_choice = (
                    # 得到对应的字符 + 概率值
                    f"{tokenizer.decode(token_id)} ({100 * token_prob:.2f}%)"  # 取百分号两位数
                )
                iteration[f"Choice {choice_idx + 1}"] = token_choice
            input_ids = torch.cat([input_ids, sorted_ids[None, 0, None]], dim=-1)  # 将概率最大的字符拼接到提示文本
            iterations.append(iteration)
        print(iterations)


def greedy_decode(ad):
    max_length = len(ad) + 30
    input_ids = tokenizer(ad, return_tensors='pt')['input_ids']
    output_greedy = model.generate(input_ids, max_length=max_length, do_sample=False, eos_token_id=None)
    res = tokenizer.decode(output_greedy[0])
    vocab = res.split('[SEP]')[1:-1]
    vocab = [v.replace(' ', '') for v in vocab]
    return vocab


def beamsearch_decode(ad):
    max_length = len(ad) + 30
    input_ids = tokenizer(ad, return_tensors='pt')['input_ids']
    output_beam = model.generate(input_ids, max_length=max_length, num_beams=3, do_sample=False, no_repeat_ngram_size=2)
    res = tokenizer.decode(output_beam[0])
    vocab = res.split('[SEP]')[1:-1]
    vocab = [v.replace(' ', '') for v in vocab]
    return vocab


def greedy_sample_decode(ad):
    max_length = len(ad) + 30
    input_ids = tokenizer(ad, return_tensors='pt')['input_ids']
    repetition_penalty = 1.1
    temperature = 1
    topk = 5
    topp = 0.95
    # 跟原始编码对齐
    output_greedy_random = model.generate(input_ids, max_length=max_length, do_sample=True,
                                          temperature=temperature, top_k=topk, top_p=topp,
                                          repetition_penalty=repetition_penalty)
    res = tokenizer.decode(output_greedy_random[0])
    vocab = res.split('[SEP]')[1:-1]
    vocab = [v.replace(' ', '') for v in vocab]
    return vocab


def beamsearch_sample_decode(ad):
    max_length = len(ad) + 30
    input_ids = tokenizer(ad, return_tensors='pt')['input_ids']
    repetition_penalty = 1.1
    num_beams = 3
    temperature = 1
    topk = 5
    topp = 0.95
    output_beamsearch_random = model.generate(input_ids, max_length=max_length, do_sample=True,
                                              num_beams=num_beams, temperature=temperature, top_k=topk,
                                              top_p=topp, repetition_penalty=repetition_penalty)
    res = tokenizer.decode(output_beamsearch_random[0])
    vocab = res.split('[SEP]')[1:-1]
    vocab = [v.replace(' ', '') for v in vocab]
    return vocab


if __name__ == '__main__':
    args = set_args()
    tokenizer = BertTokenizer.from_pretrained(args.pretrain_model_path)

    # 加载模型
    model_path = os.path.join(args.output_dir, 'model_epoch_{}'.format(9))
    model = GPT2LMHeadModel.from_pretrained(model_path)

    if torch.cuda.is_available():
        model.cuda()
    # model.half()
    model.eval()
    ad = '我女朋友是迪丽热巴'
    keyword_list = greedy_decode(ad)
    print(keyword_list)   # ['迪丽热巴', '媒体', '花絮', '三生三世', '嘉行迪丽热巴工作室']

    keyword_list = beamsearch_decode(ad)
    print(keyword_list)   # ['女明星', '媒体', '花絮', '三生三世', '快手', '空降', '虎年', '月刊']

    keyword_list = greedy_sample_decode(ad)
    print(keyword_list)   # ['热巴', '媒体', '花絮', '三生三世', 'cp', '表情包']

    keyword_list = beamsearch_sample_decode(ad)
    print(keyword_list)  # ['迪丽热巴', '媒体', '花絮', '三生三世', '嘉行迪丽热巴工作室']
```
上述代码中几个参数的介绍:
- temperature(温度参数调整解码): 相当于是预测token时候，在softmax上加一个温度参数。 如果如果temperature参数大于1。 
  整体的概率的量级都变小了，解码更多样。如果temperature参数小于1，则概率更极端。大的更大，小的更小，就接近于greedy search。
- top_k(top_k解码): 是从预测的概率中挑选出K个最有可能的下一个单词，并且仅在这K个下一个单词之间重新为它们分配概率。 也就是对他们的概率进行softmax,以这样的概率随机采样。gpt2使用的方式。
- top_p(top_p解码): top_p是设置一个累加概率的阈值。如果阈值大于top_p就停止。这前几个token就是我们的候选采样列表。然后重新分配概率，随机采这些token.
- no_repeat_ngram_size参数: 是给过去生成的token施加惩罚，这个参数如果设置2，就是如果这个词过去出现两次了，下次如果还出现，将其下一个token的概率设置为零。

# 14. mseloss中引入相关性损失
```python
from torch import nn
import torch


class MSECorrLoss(nn.Module):
    def __init__(self, p=1.5):
        super(MSECorrLoss, self).__init__()
        self.p = p
        self.mseLoss = nn.MSELoss()

    def forward(self, logits, target):
        assert (logits.size() == target.size())
        mse_loss = self.mseLoss(logits, target)   # 均方误差损失

        logits_mean = logits.mean(dim=0)
        logits_std = logits.std(dim=0)
        logits_z = (logits - logits_mean) / logits_std   # 正态分布标准化

        target_mean = target.mean(dim=0)
        target_std = target.std(dim=0)
        target_z = (target - target_mean) / target_std   # 正态分布标准化
        corr_loss = 1 - ((logits_z * target_z).mean(dim=0))   # 后面的减数 就是计算两个分布的相关系数，然后用1减。越相关损失越小。
        loss = mse_loss + self.p * corr_loss
        return loss


if __name__ == '__main__':
    logits = torch.tensor([[0.1], [0.2], [0.2], [0.8], [0.4]], dtype=torch.float32)
    label = torch.tensor([[0], [1], [0], [0], [1]], dtype=torch.float32)
    print(logits.size())   # torch.Size([5, 1])   (batch_size, 1)
    print(label.size())  # torch.Size([5, 1])   (batch_size, 1)
    loss_func = MSECorrLoss()
    loss = loss_func(logits, label)
    print(loss)   # tensor([1.9949])
```

# 15. pytorch中model.train和model.eval以及torch.no_grad()的含义
- model.train()的作用是 **启用 Batch Normalization 和 Dropout。** 如果模型中有BN层(Batch Normalization）和Dropout，需要在训练时添加model.train()。model.train()是保证BN层能够用到每一批数据的均值和方差。对于Dropout，model.train()是随机取一部分网络连接来训练更新参数。
- model.eval()的作用是 **不启用 Batch Normalization 和 Dropout。** 如果模型中有BN层(Batch Normalization）和Dropout，在测试时添加model.eval()。model.eval()是保证BN层能够用全部训练数据的均值和方差，即测试过程中要保证BN层的均值和方差不变。对于Dropout，model.eval()是利用到了所有网络连接，即不进行随机舍弃神经元。
- with torch.no_grad()则主要是用于停止autograd模块的工作，以起到加速和节省显存的作用。它的作用是将该with语句包裹起来的部分停止梯度的更新，从而节省了GPU算力和显存，但是并不会影响dropout和BN层的行为。

# 16. pytorch中张量之间的运算
```python
import torch
from einops import rearrange, repeat, reduce


def func1():
    # torch.einsum的使用
    # 1. 矩阵乘
    A = torch.randn(size=(32, 768))
    B = torch.randn(size=(768, 32))
    res = torch.einsum('ij,jk->ik', A, B)   # 矩阵乘
    print(res.size())   # torch.Size([32, 32])

    # 2. 带有batch的矩阵乘
    A = torch.randn(size=(32, 128, 768))
    B = torch.randn(size=(32, 768, 128))
    res = torch.einsum('bij,bjk->bik', A, B)
    print(res.size())   # torch.Size([32, 128, 128])

    # 3. 点乘 哈达玛乘
    A = torch.randn(size=(32, 768))
    B = torch.randn(size=(1, 768))
    res = torch.einsum('ik,jk->ik', A, B)   # 点乘
    print(res.size())   # torch.Size([32, 768])

    # 4. 对第一个维度（列）求和
    A = torch.randn(size=(32, 128, 768))
    res = torch.einsum('ijk->ik', A)
    print(res.size())   # torch.Size([32, 768])

    # 5. 维度变化
    A = torch.randn(size=(32, 128, 768))
    res = torch.einsum('ijk->ikj', A)
    print(res.size())   # torch.Size([32, 768, 128])


def func2():
    # rearrange 用法
    A = torch.randn(2, 3, 4)
    C = rearrange(A, 'b l h -> b (l h)')   # # torch.Size([2, 12]) 把后面两个维度拼在一起。
    print(C.size())   # torch.Size([2, 12])

    C = rearrange(A, 'b l h -> l (b h)')  # 交换第一和第二维度，并把它们拼在一起。
    print(C.size())    # torch.Size([3, 8])

    # 交换第一、第二维度
    C = rearrange(A, 'b l h -> l b h')
    print(C.size())

    # 拆分某个维度
    A = torch.randn(3, 12, 9)
    res = rearrange(A, 'b (m n) h -> b m n h', m=3)
    # # 将中间的维度12看作是两个维度m和n  然后指定m或者n的维度  就可以将中间维度拆成m×n的样子
    print(res.size())   # torch.Size([3, 3, 4, 9])

    # 也可以进行维度的扩展 比如在最后面多加一个维度
    a = torch.randn(3, 9, 9)
    output_tensor = rearrange(a, 'a b (c l) -> a b c l', l=1)
    print(output_tensor.size())  # # torch.Size([3, 9, 9, 1])


def func3():
    # repeat用法
    # 1. 将张量沿某个维度重复n次
    a = torch.randn(9, 9)  # 可以将其想成长度为9  词向量维度为9的一个样本
    output_tensor = repeat(a, 'l h -> b l h', b=5)
    # 将a整体重复5遍。 这里的5可以看作是batch_size
    print(output_tensor.size())  # torch.Size([5, 9, 9])

    a = torch.randn(3, 9, 9)  # 假设我们想把每条数据重复五遍
    output_tensor = repeat(a, 'b l h -> b new_axis l h', new_axis=5)
    print(output_tensor.size())  # torch.Size([3, 5, 9, 9])


def func4():
    # 沿着某个维度将其他维度平均
    a = torch.randn(3, 4, 5)  # 假设这里的维度信息为: batch_size, max_len, hidden_size
    # 我们现在要进行mean-pooling  也就是沿着第二个维度 对第三个维度进行平均
    output_tensor = reduce(a, 'b l h -> b h', 'mean')
    print(output_tensor.size())  # torch.Size([3, 5])

    output_tensor = reduce(a, 'b l h -> b h', 'max')   # max-pooling
    print(output_tensor.size())   # torch.Size([3, 5])

    # 用reduce实现卷积网络中的池化
    a = torch.randn(3, 3, 24, 24)  # 假设这里的维度信息为: batch_size, channel_size, width, height
    # this is mean-pooling with 2x2 kernel
    # image is split into 2x2 patches, each patch is averaged
    output_tensor = reduce(a, 'b c (w w1) (h h1) -> b c w h', 'mean', h1=2, w1=2)
    print(output_tensor.size())  # torch.Size([3, 3, 12, 12])


if __name__ == '__main__':
    # func1()
    # func2()
    # func3()
    func4()

```
# 17. labelsmooth的实现
```python
import torch
from torch import nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.1, reduction='mean'):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        if self.reduction == 'sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction == 'mean':
                loss = loss.mean()
        return loss * self.eps/c + (1 - self.eps) * F.nll_loss(log_preds, target, reduction=self.reduction)


if __name__ == '__main__':
    output = torch.randn(size=(5, 3))  # batch_size=5 label_nums=3
    target = torch.tensor([0, 1, 2, 0, 2], dtype=torch.long)
    loss_func = LabelSmoothingCrossEntropy()
    loss = loss_func(output, target)
    print(loss)
```
# 18. 预训练模型微调前加点噪声NoisyTune
```python
# https://aclanthology.org/2022.acl-short.76.pdf
import torch
from transformers.models.bert import BertModel

model = BertModel()
noise_lambda = 0.1   # 0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3
for name, para in model.named_parameters():
    model.state_dict()[name][:] += (torch.rand(para.size() - 0.5) * noise_lambda * torch.std(para))   # 均匀分布*noise_lambda+方差
```

# 19. 模型压缩
## 权重压缩
将float32 变为float16
```python
import torch
from transformers.models.bert import BertModel, BertConfig


if __name__ == "__main__":
    config = BertConfig.from_pretrained('./roberta_pretrain')
    model = BertModel(config=config)

    # 加载权重
    # model.load_state_dict(torch.load('./roberta_pretrain/pytorch_model.bin'), strict=False)

    # 半精度保存
    params = torch.load('./roberta_pretrain/pytorch_model.bin')   # float32
    for key in params.keys():
        params[key] = params[key].half()  # float16
    torch.save(params, './roberta_pretrain/pytorch_model_half.bin')

    # 试着加载一下
    model.load_state_dict(torch.load('./roberta_pretrain/pytorch_model_half.bin'), strict=False)
    print(model)
```

## 权重裁剪
在模型训练完成后可以考虑对冗余的权重进行裁剪，有以下几种裁剪方法：
- 按照比例随机裁剪
- 按照权重大小裁剪
https://pytorch.org/tutorials/intermediate/pruning_tutorial.html

在使用权重裁剪需要注意：
- 权重裁剪并不会改变模型的权重大小，只是增加了稀疏性；
- 权重裁剪并不会减少模型的预测速度，只是减少了计算量；
- 权重裁剪的参数比例会对模型精度有影响，需要测试和验证；

```python
import torch
import torch.nn.utils.prune as prune
from torch import nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square conv kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5x5 image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, int(x.nelement() / x.shape[0]))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def calc_sparsity(model):
    # 计算模型稀疏性
    score = 100. * float(torch.sum(model.conv1.weight == 0)) / float(model.conv1.weight.nelement())
    return "Sparsity in conv1.weight: {:.2f}%".format(score)


if __name__ == '__main__':
    model = LeNet()
    res = calc_sparsity(model)
    print(res)   # Sparsity in conv1.weight: 0.00%

    # 打印一下未修剪的conv1
    module = model.conv1
    # print(list(module.named_parameters()))
    # print(list(module.named_buffers()))   # []   没有缓冲区

    # 随机裁剪  剪掉30%的权重
    prune.random_unstructured(module, name="weight", amount=0.3)
    # print(list(module.named_parameters()))  # 其实 修剪后 参数还是不变的  有变化的是下面的缓冲区 会形成一个mask矩阵 代表剪掉哪些权重
    # print(list(module.named_buffers()))   # []   没有缓冲区

    # torch.nn.utils.prune（通过将掩码与原始参数组合）并将它们存储在属性中weight
    # print(module.weight)

    # print(model.state_dict().keys())
    '''
    odict_keys(['conv1.bias', 'conv1.weight_orig', 'conv1.weight_mask', 'conv2.weight', 'conv2.bias', 'fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias', 'fc3.weight', 'fc3.bias'])
    可以看出  加了 'conv1.weight_orig', 'conv1.weight_mask'
    '''
    res = calc_sparsity(model)
    print(res)   # Sparsity in conv1.weight: 29.63%
    
    # 另外的几种裁剪方式
    # l1_unstructured裁剪
    prune.l1_unstructured(module, name="weight", amount=0.3)
    # ln_structured裁剪
    prune.ln_structured(module, name="weight", amount=0.5, n=2, dim=0)
```

