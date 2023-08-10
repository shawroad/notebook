- [1. python中时间的处理: time和datetime](#1-python中时间的处理-time和datetime)
- [2. python起服务传输文件](#2-python起服务传输文件)
- [3. 使用nc命令进行文件传输](#3-使用nc命令进行文件传输)
- [4. linux下使用crontab设置定时任务](#4-linux下使用crontab设置定时任务)
- [5. vim中常用的tips](#5-vim中常用的tips)
- [6. python快速实现列表的交集差集并集](#6-python快速实现列表的交集差集并集)
- [7.python获取一个文件的修改日期](#7python获取一个文件的修改日期)
- [8. linux修改文件的权限](#8-linux修改文件的权限)
- [9. flask使用GET和POST两种方式传输数据](#9-flask使用get和post两种方式传输数据)
  - [GET方式](#get方式)
  - [POST的方式](#post的方式)
- [10. python中tqdm的使用详解](#10-python中tqdm的使用详解)
- [11. Faiss召回加速另外实现cos的计算](#11-faiss召回加速另外实现cos的计算)
- [12.python实现有放回抽样和无放回抽样](#12python实现有放回抽样和无放回抽样)
- [13. 编辑距离的计算](#13-编辑距离的计算)
- [14. 多个shell命令按顺序执行](#14-多个shell命令按顺序执行)
- [15. loguru的用法](#15-loguru的用法)
- [16. shell脚本监控当前程序是否在运行，否则重启](#16-shell脚本监控当前程序是否在运行否则重启)
- [17. python设定某个函数超时报错](#17-python设定某个函数超时报错)
- [18. 调用某个函数报错重试](#18-调用某个函数报错重试)
- [19. 使用numba加速python代码](#19-使用numba加速python代码)
- [20. 浅谈装饰器](#20-浅谈装饰器)
  - [函数作为参数进行传递](#函数作为参数进行传递)
  - [函数的嵌套](#函数的嵌套)
  - [实现一个装饰器](#实现一个装饰器)
- [21. joblib实现快速实现多进程多线程](#21-joblib实现快速实现多进程多线程)
  - [多进程的使用](#多进程的使用)
  - [多线程的使用](#多线程的使用)
- [22. exec函数的使用](#22-exec函数的使用)
- [23. python中map和filter以及sorted的用法](#23-python中map和filter以及sorted的用法)
- [24. faiss召回的进一步优化](#24-faiss召回的进一步优化)
- [25. python中function的partial用法](#25-python中function的partial用法)
- [26. 迭代器和迭代器对象的区别](#26-迭代器和迭代器对象的区别)
- [27. duckduckgo的使用](#27-duckduckgo的使用)
  

# 1. python中时间的处理: time和datetime

```python
  import time
  import datetime


  if __name__ == '__main__':
      cur_time = str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ', 星期' + str(datetime.datetime.now().isoweekday())
      print(cur_time)  # 2022-06-24 19:02:25, 星期:5
    
      hour_minutes_seconds = time.strftime('%H:%M:%S', time.localtime(time.time()))
      year_month_day = time.strftime('%Y-%m-%d ', time.localtime(time.time()))
      year_month_day_hour_minutes_seconds = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
      print(hour_minutes_seconds)   # 10:56:16
      print(year_month_day)   # 2021-11-18
      print(year_month_day_hour_minutes_seconds)   # 2021-11-18 10:56:16

      start_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
      print('开始时间:', start_time)   # 开始时间: 2021-11-18 10:56:16
      time.sleep(2)
      end_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
      print('结束时间:', end_time)   # 结束时间: 2021-11-18 10:56:18

      time_1_struct = datetime.datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
      time_2_struct = datetime.datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
      cost_time = time_2_struct - time_1_struct
      print('总共花费时间:', cost_time)   # 总共花费时间: 0:00:02
      
      # 得到昨天的日期
      cur_time = datetime.datetime.today()
      oneday = datetime.timedelta(days=1)
      after_day = cur_time - oneday
      after_day = datetime.datetime.strftime(after_day, '%Y%m%d')
      print(after_day)
```

# 2. python起服务传输文件

1.  发送端起服务: python -m http.server 13456     （要输出哪个文件，就在那个文件的同级目录下起服务）

2.  接收端:

   ```shell
   mac: curl -O http://xx.xx.xx.xx:13456/文件名   （文件名就是待接受的文件）
   
   linux: wget -r http://xx.xx.xx.xx:13456/文件夹名  （直接传输一个文件夹）
   
   linux: wget http://xx.xx.xx.xx:13456/文件名   （只能传输一个文件）
   ```

# 3. 使用nc命令进行文件传输

- 接收端起服务(端口号1221可以随便指定): nc -l -4 1221 | tar -zxvf - 

  - 发送端发送文件(文件和文件夹都可以,这里的file为发送的文件): tar -zcvf - file/ | nc -4 xx.xx.xx.xx 1221 

# 4. linux下使用crontab设置定时任务

  crontab -e 进入设定
  时间设定总共有: * * * * *

  - 第一个*代表minute: 区间为 0 – 59 
  - 第二个* hour: 区间为0 – 23 
  - 第三个* day-of-month: 区间为0 – 31 
  - 第四个* month: 区间为1 – 12. 1 是1月. 12是12月. 
  - 第五个* Day-of-week: 区间为0 – 7. 周日可以是0或7.

  eg: 
  - `1 * * * * /root/bin/backup.sh`   # 每小时的第一分钟执行bachup.sh脚本，注意这个脚本代码的路径问题，如果整个脚本都是相对路径，可以在最上面加一行`cd 相对路径`
  - `0 23 * * * /root/bin/backup.sh`   # 每天晚上11点0分执行 如果第一个0写成*则代表11点后一直执行。
  - `0 * * * * /root/bin/backup.sh`   # 每小时执行一次
  - `*/15 * * * * /root/bin/backup.sh` # 每15分钟执行一次
  - `*/5 9-17 * * * /root/bin/backup.sh`  # 每天9点到17点之间每隔5分钟执行一次
  - `0 0 1 * * /root/bin/backup.sh`  # 每个月的第一天执行

  crontab -l   #这个命令可以查看当前设定的所有定时任务列表

  crontab -r # 删除当前用户设定的定时任务

# 5. vim中常用的tips

1. 修改vim中tab的大小，在vim中，python的代码一般是4个空格，而默认的tab一般是8个空格，显然不方便。修改如下:

   ```shell
   在~/.vimrc文件中加入以下内容:
   set tabstop=4
   set softtabstop=4
   set shiftwidth=4
   set expandtab
   如果没有.vimrc可以创建一个。
   内容添加完后，执行source .vimrc    即可。
   ```

2. vim中左移或者右移多行代码（若不显示行，命令行设置: `set number`即可）

   ```shell
   假设我们要将10-20行的代码左移一个空格，直接进行命令行模式，输入
   : 10, 20 <
   
   若右移一个空格
   : 10, 20 >
   ```

3. 快速回到行首行末

   ```shell
   行首: shift + ^
   行末: shift + $
   ```

4. 快速回到当前代码的第一行或者最后一行

   ```shell
   第一行: gg
   最后一行: shift + g（也就是大写的G）
   ```

5. 翻看代码

   ```shell
   一个词一个词的看: w（从前往后）
   一个词一个词的看: b (从后往前)
   ```

6. 编辑代码

   ```shell
   在光标的这个点之前插入代码: i
   在光标的这个点之后插入代码: a
   在光标所在行的下面插入一行: o
   在光标所在行的上面插入一行: shift+o（也就是大写的O）
   ```

# 6. python快速实现列表的交集差集并集

最近相对两个大列表求差集，用两个for循环做，太费时间了。若直接转为集合做，则几秒结束。代码如下:
```python
list_1 = [1, 2, 5, 7, 9]
list_2 = [2, 4, 6, 9]

union = list(set(list_1) & set(list_2))
print('交集的结果:', union)    # 交集的结果: [9, 2]

intersection = list(set(list_1) | set(list_2))
print('并集的结果:', intersection)   # 并集的结果: [1, 2, 4, 5, 6, 7, 9]

minus = list(set(list_1) - set(union))
print('差集的结果:', minus)   # 差集的结果: [1, 5, 7]
```

# 7.python获取一个文件的修改日期

需求: 某个每天会更新一下某个文件,你需要在他修改后读取这个文件用于下游任务。  解决方法: 判断一个这个文件的修改日期。
```python
import os
from datetime import datetime


def modification_date(filename):
    t = os.path.getmtime(filename)
    return datetime.fromtimestamp(t)   # fromtimestamp是将float时间转为时间类型


if __name__ == '__main__':
    modify_date = modification_date('./data_v1.csv')  # 获取文件的修改日期
    time_date = datetime.now()   # 现在的时间   
    print(time_date-modify_date)   # 15 days, 3:04:24.541202   # 时间差
```

# 8. linux修改文件的权限

1. 修改一个文件的拥有权（将文件夹data下的所有文件拥有权xiaolu）: chown -R xiaolu ./data/    
2. 修改一个文件的组权限（将文件夹data下的所有文件组改为xiaolu）: chgrp -R xiaolu ./data/     
3. rw-|r--|r-- 分别代表: u所有人的权限 | g所有组的权限 | o其他人的权限。 

  - chmod u-x file1         &ensp;&ensp;  #file1拥有者去掉x权限
  - chmod g+w file1           &ensp;&ensp;  #file1拥有组添加w权限
  - chmod u-x,g+w file1         &ensp;&ensp;  #file1拥有者去掉x权，file1拥有组添加w权限
  - chmod ugo-r file2        &ensp;&ensp; #file2的用户组其他人去掉r权限
  - chmod ug+x,o-r file3       &ensp;&ensp;  #file3用户和组添加x权限，其他人去掉r权限

# 9. flask使用GET和POST两种方式传输数据

## GET方式

```python
# 服务代码
from flask import Flask, request, jsonify

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False  # 使得返回的中文能正常显示

@app.route('/predict', methods=['GET'])
def index():
    name = request.args.get('name')
    age = request.args.get('age')
    return jsonify({'name': name, 'age': age})


if __name__ == '__main__':
    app.run()


# 请求服务代码
'''
import requests

if __name__ == '__main__':
    name = 'xl'
    age = '21'
    url_predict = 'http://127.0.0.1:5000/predict?name={}&age={}'.format(name, age)
    res = requests.get(url_predict)
    print(res.json())
'''
```

## POST的方式

```python
# 服务代码
from flask import Flask, request, jsonify

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False  # 使得返回的中文能正常显示


@app.route('/predict', methods=['POST'])
def index():
    name = request.form.get('name')
    age = request.form.get('age')
    return jsonify({'name': name, 'age': age})


if __name__ == '__main__':
    app.run()


# 请求代码
"""
import requests

if __name__ == '__main__':
    url_predict = 'http://127.0.0.1:5000/predict'
    res = requests.post(url_predict, data={'name': 'xiaolu', 'age': 18}).json()
    print(res)
"""
```

# 10. python中tqdm的使用详解

- iterable：可迭代的对象，在手动更新时不需要进行设置。
- desc：字符串，左边进度条描述文字。
- total：总的项目数。leave：布尔值，迭代完成后是否保留进度条。
- file：输出指向位置，默认是终端, 一般不需要设置。
- ncols：调整进度条宽度，默认是根据环境自动调节长度，如果设置为0，就没有进度条，只有输出的信息。
- unit：描述处理项目的文字，默认是'it'，例如: 100 it/s，处理照片的话设置为'img' ，则为 100 img/s。
- unit_scale：自动根据国际标准进行项目处理速度单位的换算，例如 100000 it/s >> 100k it/s。

```python
from tqdm import tqdm
from tqdm import trange
from time import sleep
# pip install tqdm==4.64.0
import time
from tqdm.contrib import tenumerate, tzip

if __name__ == '__main__':
    # 随机生成一批需要遍历的数据
    data = ['h', 'e', 'l', 'l', 'o']

    # 制作tqdm对象
    pbar = tqdm(data, colour='yellow')   # color可以指定进度条的颜色

    for char in pbar:
        pbar.set_description("正在迭代 {}".format(char))   # 在进度条前面定义一段文字
        sleep(0.25)

    # 嵌套的tqdm
    for i in trange(3, desc='outer loop'):
        for j in trange(100, desc='inner loop', leave=False):
            sleep(0.01)
         
    data1 = [i for i in range(100)]
    data2 = [i for i in range(100)]

    # tqdm用在zip上
    result = []
    for d1, d2, in tzip(data1, data2):
        time.sleep(0.01)
        result.append(d1 + d2)

    # tqdm用在enumerate
    for i, d in tenumerate(data1):
        time.sleep(0.01)
```

# 11. Faiss召回加速另外实现cos的计算

Faiss召回，有很多优化的方面，用batch召回
```python
import faiss
import time
import numpy as np
from tqdm import tqdm
from faiss import normalize_L2


def single_search():
    index = faiss.IndexFlatL2(d)   # 建立索引
    index.add(xb)   # 将向量加入

    start_time = time.time()
    for i in tqdm(range(5000)):
        k = 1
        D, I = index.search(xb[i * 1:(i + 1) * 1], k=k)
    end_time = time.time()
    cost_time = end_time - start_time
    print('总耗时:{}, 平均每条耗时:{}'.format(cost_time, cost_time/5000.))


def batch_search():
    start_time = time.time()
    index = faiss.IndexFlatL2(d)   # 建立索引
    # normalize_L2(xb)
    index.reset()   # 这里重置一下  重新加入向量
    index.add(xb)   # 将向量加入

    for i in tqdm(range(500)):
        # batch=10
        k = 1
        D, I = index.search(xb[i * 10:(i + 1) * 10], k=k)
    end_time = time.time()
    cost_time = end_time - start_time
    print('总耗时:{}, 平均每条耗时:{}'.format(cost_time, cost_time / 5000.))


def use_ivf_speed_search():
    # 注意两个参数的意义
    # nlist: 代表划分成多少个单元格
    # nprobe: 代表每次search时在几个单元格search
    nlist = 2500
    quantizer = faiss.IndexFlatL2(d)
    quantizer.reset()
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
    index.train(xb)   # 要进行训练一下  可以理解这里就是先对向量进行聚类
    index.add(xb)

    start_time = time.time()
    for i in tqdm(range(500)):
        # batch=10
        k = 1
        D, I = index.search(xb[i * 10:(i + 1) * 10], k=k)
    end_time = time.time()
    cost_time = end_time - start_time
    print('总耗时:{}, 平均每条耗时:{}'.format(cost_time, cost_time / 5000.))


def use_cos_search():
    nlist = 2500
    quantizer = faiss.IndexFlatL2(d)

    # 1. 对向量库中的向量、查询向量分别做归一化
    normalize_L2(xb)
    normalize_L2(xq)

    # 2. 调整成内积的计算方式
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)  # 注意 这里换成内积
    
    # 如果想用GPU
    index = faiss.index_cpu_to_all_gpus(index)   # 把索引迁到gpu上

    index.train(xb)   # 要进行训练一下  可以理解这里就是先对向量进行聚类
    index.add(xb)

    start_time = time.time()
    for i in tqdm(range(500)):
        # batch=10
        k = 1
        D, I = index.search(xb[i * 10:(i + 1) * 10], k=k)
    end_time = time.time()
    cost_time = end_time - start_time
    print('总耗时:{}, 平均每条耗时:{}'.format(cost_time, cost_time / 5000.))


if __name__ == '__main__':
    np.random.seed(520)

    nb, d = 200000, 768   # 有20w向量，每个向量维度是768维。
    nq = 5000   # 有5k个查询向量

    # 1. 先生成向量库 20w x 768
    xb = np.random.random((nb, d)).astype('float32')
    xb[:, 0] += np.arange(nb) / 1000.   # 加些扰动

    # 2. 再生成查询向量 5k x 768
    xq = np.random.random((nq, d)).astype('float32')
    xq[:, 0] += np.arange(nq) / 1000.

    # 单条查询
    single_search()

    # 按batch进行查询
    batch_search()

    # 使用IndexIVFFlat 再进行加速   即先聚类 在索引
    use_ivf_speed_search()

    # 这里在注意一下  如果想实现cos的计算
    # 首先要对向量归一化 然后将计算方式调整成内积
    use_cos_search()
```

# 12.python实现有放回抽样和无放回抽样

```python
import random

if __name__ == "__main__":
    temp = [i for i in range(10)]

    # 1. 进行有放回的采样
    temp1 = random.choices(temp, k=5)  # 长度为k的list，有放回采样
    print(temp1)   # [2, 3, 7, 7, 4] 可以看出某个值是可以被采样出多次的。因为是没抽一次 放回 继续抽 直到抽够5次

    # 2. 无放回采样
    temp2 = random.sample(temp, k=5)  # 长度为k的list，无放回采样
    print(temp2)   # [8, 0, 2, 5, 1]  无论你抽几次  都不可能同一个数字  抽出两次

```

# 13. 编辑距离的计算

```python
# pip install python-Levenshtein
import Levenshtein


if __name__ == '__main__':
    # 1. 编辑距离
    texta = '钢铁是怎样炼成的'
    textb = '钢铁是怎样练成的呢'
    print(Levenshtein.distance(texta, textb))   # 2  编辑距离: 炼->练, 加呢

    # 2. 汉明距离  两字符串要登场，比较对应位置上字符不同的个数
    str1 = '不该拿'
    str2 = '不好吗'
    print(Levenshtein.hamming(str1, str2))

    # 3. 计算莱文斯坦比。计算公式  r = (sum – ldist) / sum,
    # 其中sum是指str1 和 str2 字串的长度总和，ldist是类编辑距离。
    # 注意这里是类编辑距离，在类编辑距离中删除、插入依然+1，但是替换+2
    str1 = '你爱我吗'
    str2 = '你爱谁'
    print(Levenshtein.ratio(str1, str2))   #
    # (7-3)/7 = 0.5714285714285714
```

# 14. 多个shell命令按顺序执行

有时在训练模型的结束后，需要做额外的一些处理，将其写到一个shell脚本中，需要控制前后顺序，即模型执行完后，才去执行后处理

实验: 写三个python脚本，睡眠时间不一致，但是我们要按顺序打印出来

demo1
```python
import time
if __name__ == '__main__':
    time.sleep(5)
    s = time.strftime('%H:%m:%S', time.localtime())
    print('第一个demo1执行完毕...', s)
```

demo2
```python
import time
if __name__ == '__main__':
    time.sleep(3)
    s = time.strftime('%H:%m:%S', time.localtime())
    print('第二个demo2执行完毕...', s)
```

demo3
```python
import time
if __name__ == '__main__':
    time.sleep(1)
    s = time.strftime('%H:%m:%S', time.localtime())
    print('第三个demo3执行完毕...', s)
```

shell脚本start.sh执行
```shell
nohup python demo1.py > log1.log 2>&1 &
PID=$!; wait ${PID}
nohup python demo2.py > log2.log 2>&1 &
PID=$!; wait ${PID}
nohup python demo3.py > log3.log 2>&1 &
```

执行: nohup start.sh > temp.log 2>&1 &

# 15. loguru的用法

```python
from loguru import logger
# pip install loguru
# 和logging比，好处就是不用配置  方便快捷。


if __name__ == '__main__':
    # 1. 打印出几种不同类型的信息
    logger.debug('this a debug message')   # 相当于打印出这条信息
    logger.info('this a common message')
    logger.warning('this a warning message')
    logger.critical('this a critical message')

    # 2. 先定义要将log保存到哪里  然后在开始记录  则会自动保存
    logger.add('./log/log.log', rotation="200kb", compression='zip')
    logger.add("file_{time}.log", rotation="00:00")   # 每天晚上落一个日志
    logger.add("file_{time}.log", rotation="1 days")  # 每隔一天。

    # rotation可以设定每个log保存多大的日志  到达以后，则会按时间大小写好。
    # compression 可以将日志进行压缩。  compression='zip'
    for i in range(1000):
        logger.info('这是要保存的第一条信息')
        logger.warning('这是要保存的第二条信息')

```
# 16. shell脚本监控当前程序是否在运行，否则重启

```shell
# 切到项目文件夹
cd /usr/home/project/

# 看当前的这个文件是否在运行  如果在运行 process_num肯定大于等于1  多进程的话 就会大于1
process_num=`ps -ef | grep main.py | grep -v grep | wc -l`   
if [ $process_num -ge 1 ]; then
    echo "本地服务运行正常"
else
    # 否则重启
    sh xxxx.sh 
    exit
fi
```

# 17. python设定某个函数超时报错

```python
import time
import timeout_decorator
# pip install timeout-decorator


@timeout_decorator.timeout(5)
def mytest():
    print("Start")
    for i in range(1, 6):
        time.sleep(1)
        print("{} seconds have passed".format(i))

if __name__ == '__main__':
    mytest()
```

# 18. 调用某个函数报错重试

```python
import random
# pip install tenacity
from tenacity import retry


@retry
def demo_func1():
    a = random.random()
    print(a)

    if a >= 0.1:   # 随机生成的数 如果大于0.1则进行异常的跑出
        raise Exception


if __name__ == '__main__':
    demo_func1()   # 如果上述函数报错  则重新调用  直到调用成功为止。 
```

这个工具主要用在web请求上，如果在对web请求时，发现请求出错了，我们可以再次调用。 

但有时候可能服务端真的挂了，总不可能一直尝试吧。因此，我们可以指定尝试的次数，超过这个次数，还出现异常，则抛出。
```python
import random
from tenacity import retry, stop_after_attempt


@retry(stop=stop_after_attempt(3))   # 最大尝试三次
def demo_func1():
    a = random.random()
    print(a)

    if a >= 0.1:   # 随机生成的数 如果大于0.1则进行异常的跑出
        raise Exception


if __name__ == '__main__':
    demo_func1()   # 如果三次都错误  则最后进行异常的抛出
```

# 19. 使用numba加速python代码

numba中的jit可以是代码只编译一次。大大加快了执行的速度。这里加速肯定是对那些代码不改变的函数。如果改变，肯定又得重新编译。

```python
'''
jit： just in time compilation  只编译一次，对那些不做改变函数可以使用
'''
from numba import jit
# pip install numba -i https://pypi.douban.com/simple/
import random
import time


def calc_pi(n=1000):
    # 标准计算圆面积
    acc = 0
    for i in range(n):
        x = random.random()
        y = random.random()
        if (x ** 2 + y ** 2) < 1.0:
            acc += 1
    return 4 * acc / n


# @jit()
@jit(nopython=True)
def calc_pi_jit(n=1000):
    # 使用numba中的jit进行加速   # 装饰器就是函数包函数
    acc = 0
    for i in range(n):
        x = random.random()
        y = random.random()
        if (x ** 2 + y ** 2) < 1.0:
            acc += 1
    return 4 * acc / n


if __name__ == '__main__':
    n = 10000000
    s = time.time()
    pi = calc_pi(n)
    e = time.time()
    print(pi)
    print('执行时间:', e - s)    # 执行时间: 3.3722853660583496

    s = time.time()
    pi = calc_pi_jit(n)
    e = time.time()
    print(pi)
    print('执行时间:', e - s)  # 执行时间: 0.43329310417175293

```

# 20. 浅谈装饰器

## 函数作为参数进行传递

```python
def double(x):
    return x * 2


def triple(x):
    return x * 3


def calc_num(func, x):
    print(func(x))


if __name__ == '__main__':
    calc_num(double, 3)   
    calc_num(triple, 3)
    # 输出:
    # 6
    # 9
    # 可以看出 可以将函数作为参数进行传递。类比c++中的函数指针
```

## 函数的嵌套

```python

def get_multiple_func(n):
    def multiple(x):
        return n * x
    return multiple


if __name__ == '__main__':
    double = get_multiple_func(2)
    triple = get_multiple_func(3)
    # 以上的两个调用返回的都是函数

    # 接着调用
    print(double(3))  # 才是真正调用内部的函数
    print(triple(3))
    # 输出:
    # 6
    # 9
```

## 实现一个装饰器

```python
import time


# 一般意义上的装饰器可以认为输入是函数 输出也是函数
def timeit(f):
    def wrapper(*args, **kwargs):
        '''
        这里之所以要传*args和**kwargs就是为了通用性
        不管你这个函数有多少个参数 都不会影响我装饰器中
        计算时间的任务
        '''
        start = time.time()
        ret = f(*args, **kwargs)
        print('当前函数花费时间:', time.time() - start)
        return ret
    return wrapper


@timeit
def my_func(x):
    return x * 2


@timeit
def add(x, y):
    return x + y


if __name__ == '__main__':
    print(my_func(10))
    print('*'*50)
    print(add(1, 1))
    '''
    输出:
    当前函数花费时间: 1.1920928955078125e-06
    20
    **************************************************
    当前函数花费时间: 9.5367431640625e-07
    2
    '''
    # 结论 在函数a上面加一个装饰器b   调用其实类似于b(a)
```

# 21. joblib实现快速实现多进程多线程
**密集型计算推荐使用多进程。密集型IO推荐使用多线程。**
## 多进程的使用
```python
import time
from joblib import Parallel, delayed


def run_task(a, b):
    time.sleep(1)
    return a + b


if __name__ == '__main__':
    # 普通调用
    s = time.time()
    results = []
    for i in range(20):
        results.append(run_task(i, i))
    e = time.time()
    print("结果为:", results, '花费时间:', e - s)
    # 结果为: [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38] 花费时间: 20.055760145187378

    s = time.time()
    # 注意下面这种n_jobs=4 指的是多进程执行
    results = (Parallel(n_jobs=4)(delayed(run_task)(i, i) for i in range(20)))
    e = time.time()
    print('总共花费时间:',  e - s)
    for i in results:
        print(i, end=', ')
    # 结果为:
    # 总共花费时间: 6.1908791065216064
    # 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38,
```
## 多线程的使用
```python
import time
import requests
from joblib import Parallel, delayed


def run_task(url):
    response = requests.get(url)
    return response.status_code


if __name__ == '__main__':
    url = 'https://www.baidu.com'
    # 普通调用
    s = time.time()
    results = []
    for i in range(20):
        results.append(run_task(url))
    e = time.time()
    print("结果为:", results, '花费时间:', e - s)
    # 结果为: [200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200] 
    # 花费时间: 6.937748908996582

    s = time.time()
    # 显示指定线程  线程可以开很多   后面那个url是传的参数 可以传多个参数。
    results = (Parallel(n_jobs=20, backend='threading')(delayed(run_task)(url) for i in range(20)))
    e = time.time()
    print('结果为:', results, '花费时间:', e - s)
    # 结果为: [200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200] 
    # 花费时间: 0.5266118049621582
```

# 22. exec函数的使用
exec可以用来执行一段函数代码。这个函数代码是一个字符串。比eval功能更强大。

假设有一个文件file.txt,内容如下:
```txt
import random

if __name__ == '__main__':
    data = [random.randint(0, 10) for i in range(10)]
    data.sort()
    print(data)
```

接下来，写一个python脚本，读file.txt文件内容，然后用exec执行。
```python
with open('file.txt', 'r', encoding='utf8') as f:
    content = f.read()

exec(content)
```

输出结果: [1, 1, 1, 2, 4, 6, 9, 10, 10, 10]

# 23. python中map和filter以及sorted的用法
```python
def convert_int(item):
    temp = int(item)
    return temp


def find_even(item):
    # 找偶数
    if item % 2 == 0:
        return True
    else:
        return False


if __name__ == '__main__':
    data = ['12', '341', '244']
    res = map(convert_int, data)
    res = [i for i in res]
    # print(res)   # [12, 341, 244]

    res = filter(find_even, res)
    res = [i for i in res]
    # print(res)   # [12, 244]

    # 对字典或者列表排序
    dict1 = {'score1': 12, 'score2': 11, 'score3': 31, 'score4': 4}
    dict1 = sorted(dict1.items(), key=lambda x: x[1])
    # print(dict1)  # [('score4', 4), ('score2', 11), ('score1', 12), ('score3', 31)]

    list1 = [[1, 12, 32], [23, 12, 22], [2, 12, 42], [1, 3, 3]]
    list1 = sorted(list1, key=lambda x: x[2], reverse=True)
    # print(list1)   # [[2, 12, 42], [1, 12, 32], [23, 12, 22], [1, 3, 3]]
```

# 24. faiss召回的进一步优化
```python
import faiss
import numpy as np


def batch_search():
    # 随机产生一个向量库
    nb = 300000  # 向量库的大小
    xb = np.random.random((nb, d)).astype('float32')
    xb[:, 0] += np.arange(nb) / 1000.

    # 随机产生100个query向量
    nq = 100  # 用这100个向量进行检索
    xq = np.random.random((nq, d)).astype('float32')
    xq[:, 0] += np.arange(nq) / 1000.

    # 如果想计算cos  则是归一化 再内积 就是cos相似度
    faiss.normalize_L2(xb)  # 原地归一化 无返回值
    print(faiss_index.is_trained)

    faiss_index.train(xb)  # 训练
    print(faiss_index.is_trained)
    faiss_index.add(xb)  # 训练完后  再将所有向量加入fiass中

    faiss_index.nprobe = 10  # 每次检索 选取10个维诺空间
    k = 10
    xq = xq.tolist()
    batch_size = 16  # 批量检索  每次传16个
    n = len(xq) // batch_size
    for i in range(n + 1):
        cur_q = xq[i * batch_size: (i + 1) * batch_size]
        vecs = np.array(cur_q, dtype='float32')
        faiss.normalize_L2(vecs)
        dis, ind = faiss_index.search(vecs, k)

        for prob_list, index_list in zip(dis, ind):
            print(index_list)
            print(prob_list)
            print('*' * 100)


def direct_search(faiss_index):
    # 有新的向量直接入库  无需训练  我们已经通过大批量的数据 训练了faiss的索引

    # 在训练好的索引内加100万个向量
    nb = 1000000  # 向量库的大小
    xb = np.random.random((nb, d)).astype('float32')
    xb[:, 0] += np.arange(nb) / 1000.
    # np.random.shuffle(xb)   # 为了看出检索的效果
    faiss.normalize_L2(xb)  # 原地归一化 无返回值
    faiss_index.add(xb)  # 训练完后  再将所有向量加入fiass中

    # 随机产生100个query向量
    nq = 100  # 用这100个向量进行检索
    xq = np.random.random((nq, d)).astype('float32')
    xq[:, 0] += np.arange(nq) / 1000.

    faiss_index.nprobe = 10  # 每次检索 选取10个维诺空间
    k = 10
    xq = xq.tolist()
    batch_size = 16  # 批量检索  每次传16个
    n = len(xq) // batch_size
    for i in range(n + 1):
        cur_q = xq[i * batch_size: (i + 1) * batch_size]
        vecs = np.array(cur_q, dtype='float32')
        faiss.normalize_L2(vecs)
        dis, ind = faiss_index.search(vecs, k)

        for prob_list, index_list in zip(dis, ind):
            print(index_list)
            print(prob_list)
            print('*' * 100)


if __name__ == '__main__':
    d = 512   # 向量维度
    np.random.seed(1234)

    # 初始化faiss
    nlist = 100   # 将向量库分割成多少维诺空间
    quantizer = faiss.IndexFlatL2(d)  # the other index
    faiss_index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)   # 内积方式

    batch_search()   # 这种是每次添加新向量  就训练 肯定太耗资源了

    # 我们这里可以先用大批量的向量训练 把索引库构建起来  以后就直接加向量就行
    # 随机产生一个向量库
    nb = 300000  # 向量库的大小
    xb = np.random.random((nb, d)).astype('float32')
    xb[:, 0] += np.arange(nb) / 1000.
    # 如果想计算cos  则是归一化 再内积 就是cos相似度
    faiss.normalize_L2(xb)  # 原地归一化 无返回值
    print(faiss_index.is_trained)
    faiss_index.train(xb)  # 训练
    print(faiss_index.is_trained)
    direct_search(faiss_index)

```

# 25. python中function的partial用法
```python
import requests
from functools import partial


def get_baidu(method, url):
    response = requests.request(method=method, url=url)
    return response.content


# 最近在看paddle的一些官方案例时，反复用到partial  所以就来熟悉一下
if __name__ == "__main__":
    '''
    functools.partial（func，*args， **关键字）
    返回一个新的部分对象，当被调用时，其行为类似于使用位置参数args和关键字参数关键字调用的func。
    如果为调用提供了更多参数，则将它们附加到args。如果提供了其他关键字参数，则它们会扩展和覆盖关键字。
    简单说就是把一个函数, 和该函数所需传的参数封装到一个class。 'functools.partial'的类中, 简化以后的调用方式
    '''

    # demo1: 获取百度的html内容
    res = get_baidu("get", "https://www.baidu.com")  # 正常调用
    print(res)

    # 将函数和参数封装到一个指定变量名中,下次执行直接调用加()
    getBaidu = partial(get_baidu, "get", "https://www.baidu.com")
    print(type(getBaidu))    # <class 'functools.partial'>
    res = getBaidu()
    print(res)
```

# 26. 迭代器和迭代器对象的区别
```python
def get_hamburgers_return(qty):
    hamburgers = ["汉堡" for h in range(qty)]
    return hamburgers


def get_hamburgers_yield(qty):
    hamburgers = ("汉堡" for h in range(qty))
    for h in hamburgers:
        yield h


if __name__ == '__main__':
    hamburgers_return = get_hamburgers_return(1000000)
    hamburgers_yield = get_hamburgers_yield(100000)

    print(f"1000000个汉堡占用的普通函数空间 {hamburgers_return.__sizeof__()}。")
    # 1000000个汉堡占用的普通函数空间 8697440。
    print(f"1000000个汉堡占用的生成器空间 {hamburgers_yield.__sizeof__()}。")
    # 1000000个汉堡占用的生成器空间 96。

    '''
    ["汉堡" for h in range(100)]是列表，是一个可迭代对象。
    ("汉堡" for h in range(100))是一个迭代器。
    
    两句话复习下迭代器和可迭代对象的差别，
    「迭代器是实现迭代器协议的对象，该协议由方法__iter__()和__next__()组成。」
    「可迭代对象则只有__iter__()方法。」
    '''

    # 如果想取出每个数据  都是用for
    # for i in hamburgers_return:
    #     print(i)
    # for i in hamburgers_yield:
    #     print(i)

```

# 27. duckduckgo的使用
```python
from duckduckgo_search import ddg_suggestions
from duckduckgo_search import ddg_translate, ddg, ddg_news


if __name__ == '__main__':
    # 1. 获取对应的词条
    print('推荐词条:', ddg_suggestions("周杰伦"))
    # 推荐词条: [{'phrase': '周杰伦'}, {'phrase': '周杰伦概念股首日大涨23%'}, {'phrase': '周杰伦郭艾伦合唱稻香'},
    # {'phrase': '周杰伦下厨做情人节大餐'}, {'phrase': '周杰伦演唱会'}, {'phrase': '周杰伦歌曲'}, {'phrase': '周杰伦歌曲免费听'},
    # {'phrase': '周杰伦演唱会2023'}]

    # 2. 翻译
    print("翻译: \n", ddg_translate("西安是世界古都。", to="en"))
    #  {'detected_language': 'zh-Hans',
    #   'translated': "Xi'an is the ancient capital of the world.",
    #   'original': '西安是世界古都。'}

    # 3. 搜索网页
    r = ddg("北京水灾", max_results=5)
    for page in r:
        print("page: \n", page, "\n")
        # {'title': '局部地区洪水等次生灾害风险仍较大 北京暴雨最新动态一览',
        #  'href': 'http://health.people.com.cn/n1/2023/0801/c14739-40048238.html',
        #  'body': '7月31日下午，网传持续强降雨致卢沟桥坍塌。 经核实，因为永定河分洪，造成了卢沟桥西侧的小清河桥部分坍塌，并非卢沟桥塌陷。
        #           因持续降暴雨，铁路线路封闭，K396、K1178、Z180三辆列车被长时间滞留进京途中。 其中K1178次列车滞留在门头沟区沿河城附
        #           近，由于所在位置地势较高，旅客就地在车上等待救援。 此外，7月31日，北京房山区升级发布地质灾害气象风险红色预警；北京
        #           地铁7号线7月31日晚延长运营1小时，以便做好夜间北京西站旅客接驳服务；7月31日自即时起至另有通知时止，地铁大兴机场线全
        #           线停运；北京丰台小清河桥局部垮塌致五车坠河，初步判断无人员落水。 据北京市防汛抗旱指挥部消息，截至8月1日6时，此轮强
        #           降雨已经造成11人遇难，包括门头沟区4人、昌平区4人、房山区2人、海淀区1人。'}

    # 4. 搜索新闻
    print("搜索新闻:\n", ddg_news("北京水灾", safesearch='Off', time='d', max_results=5))
    # [{'date': '2023-08-09T12:04:00',
    #   'title': '北京大洪水，《明史》和《清史》有记录，有的暴雨持续1个月',
    #   'body': '根据最权威和最详实的史料来源，例如明清时期的《明史》《清史》《清史稿》等，新中国成立后的《北京市志》《北京市年鉴》等，
    #            记录了北京历史上发生过的各种水灾的时间、规模、影响和应对措施。',
    #   'url': 'https://www.sohu.com/a/710330992_121772077',
    #   'image': None,
    #   'source': '搜狐'},
    #  {'date': '2023-08-10T03:55:00',
    #   'title': '暴雨过后的环北京经济圈',
    #   'body': '分析一下经济数据可以看到南北方县城的差距，在环北京的这16个县城里，这次水灾最为严重的涿州市2022年GDP总量为400亿元。
    #            在环北京16县市中排名第三，仅次于天津武清区890亿和廊坊北三县中三河市的610亿元。 然而无论是三河还是涿州，在全国都很难
    #            排进百强县的行列。要是昆山、江阴那些江苏经济强县相比，经济体量的差距几乎要在达到10倍。',
    #   'url': 'https://www.163.com/dy/article/IBPDICQD0519D9A7.html',
    #   'image': None,
    #   'source': '网易'}]

```
