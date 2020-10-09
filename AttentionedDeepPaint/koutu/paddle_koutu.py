#coding: utf-8
import os
import paddlehub as hub

# 2、加载模型
humanseg = hub.Module(name='deeplabv3p_xception65_humanseg')

# 3、获取文件列表
# 图片文件的目录
path = '/home/SENSETIME/zhangjunwei/data/zhangjw/project/AttentionedDeepPaint/koutu/imgs/'
# 获取目录下的文件
files = os.listdir(path)
print("file is: ",files)

# 用来装图片的
imgs = []
# 拼接图片路径
for i in files:
    imgs.append(path + i)
    print("imgs is: ",imgs)
#抠图
results = humanseg.segmentation(data={'image':imgs})