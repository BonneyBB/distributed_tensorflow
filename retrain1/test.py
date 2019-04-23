# utf-8
# 使用incepton-v3做各种图像的识别
import tensorflow as tf
import os
import numpy as np
import re
from PIL import Image
import matplotlib.pyplot as plt

lines = tf.gfile.GFile('retrained_labels.txt').readlines()
uid_to_human = {}
# 一行一行的读取数据
for uid, line in enumerate(lines):
    # 去掉换行符
    line = line.strip('\n')
    uid_to_human[uid] = line


def id_to_string(id):
    if id not in uid_to_human:
        return ''
    return uid_to_human[id]


# 创建一个图来存放google训练好的模型
with tf.gfile.FastGFile('retrained_graph.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')


with tf.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    # 遍历目录
    for root, dirs, files in os.walk('imgs'):
        for file in files:
            # 载入图片
            image_data = tf.gfile.FastGFile(os.path.join(root, file), 'rb').read()
            predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data}) # 图片格式是jpg
            predictions = np.squeeze(predictions) #  把结果转化为1维数据
            # 打印图片路径和名称
            image_path = os.path.join(root, file)
            print(image_path)
            # 显示图片
            img = Image.open(image_path)
            plt.imshow(img)
            plt.axis('off')
            plt.show()

            # 排序
            top_k = predictions.argsort()[::-1]
            for node_id in top_k:
                # 获取分类名称
                human_string = id_to_string(node_id)
                # 获取该分类的置信度
                score = predictions[node_id]
                print('%s (score=%.5f)' % (human_string, score))
            print("该模型判定该图为:" + id_to_string(top_k[0]))
            print()