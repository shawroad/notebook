- [1. 解决pandas进行csv文件保存时出现乱码的问题](#1-解决pandas进行csv文件保存时出现乱码的问题)
- [2. pandas中对列对行的操作另外对数据去重: map()、apply()、drop_duplicates()](#2-pandas中对列对行的操作另外对数据去重-mapapplydrop_duplicates)
- [3. pandas按某列数据进行排序、将pandas自动隐藏的列显示出来](#3-pandas按某列数据进行排序将pandas自动隐藏的列显示出来)
- [4. pandas针对某列的nan值删除以及位置索引的用法。](#4-pandas针对某列的nan值删除以及位置索引的用法)
- [5. Jieba加入自定义的词进行分词](#5-jieba加入自定义的词进行分词)
- [6. pandas使用concat对数据拼接并用loc对数据索引读取](#6-pandas使用concat对数据拼接并用loc对数据索引读取)
- [7. 对dataframe数据按行进行shuffle操作](#7-对dataframe数据按行进行shuffle操作)
- [8. 使用opencv操作视频的帧](#8-使用opencv操作视频的帧)
- [9. 将图片通过base64转为字符串](#9-将图片通过base64转为字符串)


#  1. 解决pandas进行csv文件保存时出现乱码的问题

- 第一种: data.to_csv(file_name, encoding='utf8')
- 第二种: data.to_csv(file_name, encoding='utf_8_sig')

区别: 

1.  ”utf-8“ 是以字节为编码单元,它的字节顺序在所有系统中都是一样的,没有字节序问题,因此它不需要BOM,所以当用"utf-8"编码方式读取带有BOM的文件时,它会把BOM当做是文件内容来处理, 也就会发生类似上边的错误.
2. “utf-8-sig"中sig全拼为 signature 也就是"带有签名的utf-8”, 因此"utf-8-sig"读取带有BOM的"utf-8文件时"会把BOM单独处理,与文本内容隔离开,也是我们期望的结果.

# 2. pandas中对列对行的操作另外对数据去重: map()、apply()、drop_duplicates()

  - map()主要作用series中的每个元素
  - apply()将一个函数作用于DataFrame中的每个行或者列  但是也可以做map()做的事情

  ```python
  import pandas as pd
  import re


  def text_normal_l1(text):
      # 对数据进行简单清洗
      rule_url = re.compile(
          '(https?://)?(www\\.)?[-a-zA-Z0-9@:%._+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b([-a-zA-Z0-9()@:%_+.~#?&/=]*)'
      )
      rule_legal = re.compile('[^\\[\\]@#a-zA-Z0-9\u4e00-\u9fa5]')
      rule_space = re.compile('\\s+')
      text = str(text).replace('\\n', ' ').replace('\n', ' ').strip()
      text = rule_url.sub(' ', text)
      text = rule_legal.sub(' ', text)
      text = rule_space.sub(' ', text)
      return text.strip()

  def text_normal_l1_v2(row):
      text = row['text']
      # 对数据进行简单清洗
      rule_url = re.compile(
          '(https?://)?(www\\.)?[-a-zA-Z0-9@:%._+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b([-a-zA-Z0-9()@:%_+.~#?&/=]*)'
      )
      rule_legal = re.compile('[^\\[\\]@#a-zA-Z0-9\u4e00-\u9fa5]')
      rule_space = re.compile('\\s+')
      text = str(text).replace('\\n', ' ').replace('\n', ' ').strip()
      text = rule_url.sub(' ', text)
      text = rule_legal.sub(' ', text)
      text = rule_space.sub(' ', text)
      return text.strip()

  if __name__ == '__main__':
      # 1. 瞎编一些数据
      data = {'id': [1, 2, 3, 4, 5],
              'text': ['fdas你是？fda8*我下换呢。。24ad90-',
                       't3r5qr89080noif打你哦哦琴女31390》《？《',
                       '&*()&dnfkaln你大分类进发了https://baidu.com 咖啡机  fjadifjad',
                       '<>?><faniovjalkdjfdNIONIOJO大据了解9））',
                       '4859&*)*)))njkfdankldnfd即可厉害'
                      ]
              }
      data = pd.DataFrame(data)
     
      # 2. 针对text列进行清洗。
      # 第一种方式
      # data['text'] = data['text'].apply(text_normal_l1)
  
      # 第二种方式
      # data['text'] = data['text'].map(text_normal_l1)
      
      # 第三种方式
      # data.loc[:, 'text'] = data.apply(text_normal_l1_v2)   # 直接传进去一行  然后再函数里面针对列处理。

      # 3. 对id列标签分别加100
      # 第一种方式
      # data['id'] = data['id'].map(lambda x: x+100)

      # 第二种方式
      # data['id'] = data['id'].apply(lambda x: x+100)

      # 4. 如果要将id转为字符串然后和text连起来 就只能apply做了  记住: apply默认是对列操作，如果想对行操作，指定axis=1
      data['new_col'] = data[['id', 'text']].apply(lambda x: str(x[0]) + ':' + x[1], axis=1)
      
      # 5. 对数据去重
      data.drop_duplicates(subset='text', keep='first', inplace=True)   # 如果重复 直接保留第一次出现的结果
  ```

# 3. pandas按某列数据进行排序、将pandas自动隐藏的列显示出来

  可以对数值型数据、字符串进行排序。

  ```python
  import pandas as pd


  if __name__ == '__main__':
      data = pd.DataFrame({'age': ['2', '3', '4', '1', '2', '5', '2'],
                           'name': ['fa', 'da', 'df', 'bfd', 'bg', 'hr', 'os']})

      # 可以对数值型数据排序、字符串排序
      data.sort_values('age', inplace=True)  # 按升序对年龄列排序
      data.sort_values('age', inplace=True, ascending=False)  # 按照降序对年龄列排序
      print(data)

      # 将其转为数值 排序
      data['age'] = data['age'].map(lambda x: int(x))
      data.sort_values('age', inplace=True)
      print(data)
      
      # 如果列太多，pandas会把一些列隐藏，让其显示处理
      pd.set_option("display.max_columns", None)
      data = pd.read_csv('./data_v1.csv')
      print(data.head())
  ```

# 4. pandas针对某列的nan值删除以及位置索引的用法。

  下面的例子是: 如果age中有缺失值，则直接删除所在的行。 另外用切片的方式去取数据。 

  ```python
  import pandas as pd

  df = pd.DataFrame({'name': ['z', 'x', 'd', 'e', 'f'], 'age': [2, float('nan'), 3, 32, 12]})
  print(df)
  print('*'*10)
  df.dropna(subset=['age'], inplace=True)
  print(df)

  # 位置索引
  print(df.iloc[0, 1])
  print(df.iloc[0:, 1:])
  
  # df.reset_index(inplace=True)   # 重置其索引，因为的dropna的时候 将索引弄得不连续了
  ```

# 5. Jieba加入自定义的词进行分词

```python
import jieba


if __name__ == '__main__':
    s = '我爱你，就像老师自定义词x爱大自定义词2米。'
    print('正常jieba分词结果:', jieba.lcut(s))
    # 正常jieba分词结果: ['我爱你', '，', '就', '像', '老师', '自定义词', 'x', '爱大', '自定义词', '2', '米', '。']

    # 我们将"自定义词x"和"自定义词2"分出来
    jieba.add_word("自定义词x")
    jieba.add_word("自定义词2")
    print('加入自定义词后的分词结果:', jieba.lcut(s))
    # 加入自定义词后的分词结果: ['我爱你', '，', '就', '像', '老师', '自定义词x', '爱大', '自定义词2', '米', '。']

    # 删除自定义词
    jieba.del_word("自定义词x")
    jieba.del_word("自定义词2")
    print('删除自定义词后的分词结果:', jieba.lcut(s))
    # 删除自定义词后的分词结果: ['我爱你', '，', '就', '像', '老师', '自定义词', 'x', '爱大', '自定义词', '2', '米', '。']
```

# 6. pandas使用concat对数据拼接并用loc对数据索引读取

```python
import pandas as pd

if __name__ == '__main__':
    df1 = pd.DataFrame({"sid": ["s1", "s2"], "name": ["xiaoming", "Mike"]})
    df2 = pd.DataFrame({"sid": ["s3", "s4"], "name": ["Tom", "Peter"]})
    df3 = pd.DataFrame({"address": ["北京", "深圳"], "sex": ["Male", "Female"]})

    # 将df1和df2纵向拼接
    data1 = pd.concat([df1, df2])
    data1.reset_index(drop=True, inplace=True)   # 合并以后要重置索引 不然下面用loc取数据就报错了。
    print(data1)

    # 将df1和df3横向拼接
    data2 = pd.concat([df1, df3], axis=1)
    print(data2)

    # loc按索引取数据 可以切片
    print(data1.loc[1:3, :])
```

# 7. 对dataframe数据按行进行shuffle操作

```python
import pandas as pd
from sklearn.utils import shuffle


if __name__ == '__main__':
    data = pd.read_csv('supervised_train_data_9734.csv')
    data = shuffle(data)
    data.to_csv('supervised_train_data_9734_shuffle.csv', index=False)    
```

# 8. 使用opencv操作视频的帧

1. 读取帧数、帧率等
2. 按秒保存每帧图片

```python
import cv2


def save_second_picture():
    # 1. 加载视频
    path = './baseline/VideoLabelTool/video/4300159380689400.mp4'
    videoCapture = cv2.VideoCapture(path)

    # 2. 读取每帧
    success, frame = videoCapture.read()

    # 3. 计算一下当前视频的帧率
    fps = int(videoCapture.get(cv2.CAP_PROP_FPS))  # 计算当前视频的帧率  后面就可以各多少次取一帧了

    i, j = 0, 0
    while success:
        i = i + 1
        if i % fps == 0:   # 保存
            j = j + 1
            save_path = './output/image_{}.jpg'.format(j)
            cv2.imwrite(save_path, frame)
            print('save image:', i)
        success, frame = videoCapture.read()


if __name__ == '__main__':
    path = './baseline/VideoLabelTool/video/4300159380689400.mp4'

    # 读取视频文件
    videoCapture = cv2.VideoCapture(path)

    # 通过摄像头的方式
    # videoCapture = cv2.VideoCapture(0)

    # 获取视频的总帧数
    nums = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
    print('视频的总帧数:', nums)

    # 获取视频的帧率
    fps = videoCapture.get(cv2.CAP_PROP_FPS)  # 计算视频的帧率
    print('帧率:', fps)    # 即每秒钟显示多少帧

    # 获取帧的宽 高
    width = videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print('帧的宽: {}, 高:{}'.format(width, height))

    # 假设我们要实现对某个视频 每秒钟保存一帧 如果视频为8秒 则保存8秒图片
    save_second_picture()
```

# 9. 将图片通过base64转为字符串

```python
import base64


def convert_image_to_str(source_path, save_path):
    with open(save_path, 'wb') as fw:
        with open(source_path, "rb") as fr:
            base64_data = base64.b64encode(fr.read())
            fw.write(base64_data)


def convert_str_to_image(path):
    with open(path, 'rb') as f:
        s = base64.b64decode(f.read())
    with open('xx.png', 'wb') as f:
        f.write(s)


if __name__ == '__main__':
    # 1. 将图片转为字符串
    source_path = './image_1.jpg'
    save_path = './image_1.txt'
    convert_image_to_str(source_path, save_path)

    # 2. 将字符串转为图片
    convert_str_to_image(save_path)
```
