import tensorflow as tf
import os
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt

# 遍历目录
for root, dirs, files in os.walk('imgs/'):
    for file in files:
        tf.reset_default_graph()


        def imageprepare(argv):
            im = Image.open(argv).convert('L')
            width = float(im.size[0])
            height = float(im.size[1])
            newImage = Image.new('L', (28, 28), 255)  # creates white canvas of 28x28 pixels

            if width > height:  # check which dimension is bigger
                # Width is bigger. Width becomes 20 pixels.
                nheight = int(round((20.0 / width * height), 0))  # resize height according to ratio width
                if nheight == 0:
                    # rare case but minimum is 1 pixel
                    nheight = 1
                # resize and sharpen
                img = im.resize((20, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
                wtop = int(round(((28 - nheight) / 2), 0))  # caculate horizontal pozition
                newImage.paste(img, (4, wtop))  # paste resized image on white canvas
            else:
                # Height is bigger. Heigth becomes 20 pixels.
                nwidth = int(round((20.0 / height * width), 0))  # resize width according to ratio height
                if nwidth == 0:  # rare case but minimum is 1 pixel
                    nwidth = 1
                # resize and sharpen
                img = im.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
                wleft = int(round(((28 - nwidth) / 2), 0))  # caculate vertical pozition
                newImage.paste(img, (wleft, 4))  # paste resized image on white canvas

            tv = list(newImage.getdata())  # get pixel values

            # normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
            tva = [(255 - x) * 1.0 / 255.0 for x in tv]
            return tva


        # 参数概要
        def variable_summaries(var):
            with tf.name_scope('summaries'):
                mean = tf.reduce_mean(var)  # 均值
                tf.summary.scalar('mean', mean)
                with tf.name_scope('stddev'):
                    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))  # 标准差
                tf.summary.scalar('stddev', stddev)
                tf.summary.scalar('max', tf.reduce_max(var))
                tf.summary.scalar('min', tf.reduce_min(var))
                tf.summary.histogram('histogram', var)  # 直方图


        # 初始化权值
        def weight_variable(shape):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial)


        # 初始化偏置
        def bias_variable(shape):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial)


        # 卷积层
        def conv2d(x, W):
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


        # 池化层
        def max_pool_2x2(x):
            return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


        with tf.name_scope('input'):
            # 定义两个占位符
            x = tf.placeholder(tf.float32, [None, 784])  # 28*28
            y = tf.placeholder(tf.float32, [None, 10])
            with tf.name_scope('x_image'):
                # 改变x的格式为4D的向量[batch, in_height, in_width, in_channels]
                x_image = tf.reshape(x, [-1, 28, 28, 1])

        with tf.name_scope('Conv1'):
            # 初始化第一个卷积层的权值和偏移值
            with tf.name_scope('W_conv1'):
                W_conv1 = weight_variable([5, 5, 1, 32])  # 5*5的采样窗口，32个卷积核从一个平面抽取特征
            with tf.name_scope('b_conv1'):
                b_conv1 = bias_variable(([32]))  # 每一个卷积核一个偏移值

            # 把x_image和权值向量进行卷积，再加上偏移值，然后应用relu函数进行激活
            # 第一次卷积后得到28*28，第一次池化后得到14*14
            with tf.name_scope('conv2d_1'):
                conv2d_1 = conv2d(x_image, W_conv1) + b_conv1
            with tf.name_scope('relu1'):
                h_conv1 = tf.nn.relu(conv2d_1)
            with tf.name_scope('h_pool1'):
                h_pool1 = max_pool_2x2(h_conv1)

        # 初始化第二个卷积层的权值和偏移值
        with tf.name_scope('Conv2'):
            with tf.name_scope('W_conv2'):
                W_conv2 = weight_variable([5, 5, 32, 64])
            with tf.name_scope('b_conv2'):
                b_conv2 = bias_variable([64])

            # 把h_pool1和权值向量进行卷积，再加上偏移值，然后应用relu函数进行激活
            # 第二次卷积后得到14*14， 第二次池化后得到7*7
            with tf.name_scope('Conv2d_2'):
                with tf.name_scope('W_conv2'):
                    conv2d_2 = conv2d(h_pool1, W_conv2) + b_conv2
                with tf.name_scope('relu2'):
                    h_conv2 = tf.nn.relu(conv2d_2)
                with tf.name_scope('h_pool2'):
                    h_pool2 = max_pool_2x2(h_conv2)
        # 经过上面的操作后得到 64张7*7的平面

        # 初始化第一个全连接层的权值和偏移值
        with tf.name_scope('fc1'):
            with tf.name_scope('W_fc1'):
                W_fc1 = weight_variable([7 * 7 * 64, 1024])  # 上一层有7*7*64个神经元，全连接层有1024个神经元
            with tf.name_scope('b_fc1'):
                b_fc1 = bias_variable([1024])

            # 把池化层2的输出扁平化为一维
            with tf.name_scope('h_pool2_flat'):
                h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
            # 求第一个全连接层的输出
            with tf.name_scope('wx_plus_b1'):
                wx_plus_b1 = tf.matmul(h_pool2_flat, W_fc1) + b_fc1
            with tf.name_scope('relu3'):
                h_fcl = tf.nn.relu(wx_plus_b1)

            # keep_prob用来表示神经元的输出概率
            with tf.name_scope('keep_prob'):
                keep_prob = tf.placeholder(tf.float32)
            with tf.name_scope('h_fcl_drop'):
                h_fcl_drop = tf.nn.dropout(h_fcl, keep_prob)

        # 初始化第二个全连接层
        with tf.name_scope('fc2'):
            with tf.name_scope('W_fc2'):
                W_fc2 = weight_variable([1024, 10])  # 上一层有7*7*64个神经元，全连接层有1024个神经元
            with tf.name_scope('b_fc2'):
                b_fc2 = bias_variable([10])
            with tf.name_scope('wx_plus_b2'):
                wx_plus_b2 = tf.matmul(h_fcl_drop, W_fc2) + b_fc2
            # 计算输出
            with tf.name_scope('softmax'):
                prediction = tf.nn.softmax(wx_plus_b2)

        # 交叉熵代价函数
        with tf.name_scope('cross_entropy'):
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
            tf.summary.scalar('cross_entropy', cross_entropy)

        # 优化器可以改变测试速度
        with tf.name_scope('train_step'):
            train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

        # 结果存放在一个bool型的列表中
        with tf.name_scope('acc'):
            with tf.name_scope('correct_prediction'):
                correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            # 求准确率，进行一次强制类型转换
            with tf.name_scope('accuracy'):
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # bool型转换成float32
                tf.summary.scalar('accuracy', accuracy)

        init_op = tf.global_variables_initializer()
        saver = tf.train.Saver()

        init_op = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init_op)
            saver.restore(sess, "logs/model.ckpt-3706")
            # 载入图片
            image_path = os.path.join(root, file)

            # 显示图片
            img = Image.open(image_path)
            plt.imshow(img)
            plt.axis('off')
            plt.show()

            # 打印图片名称
            print(file)
            tva = imageprepare(image_path)
            prediction = tf.argmax(prediction, 1)
            predint = prediction.eval(feed_dict={x: [tva], keep_prob: 1.0})
            print('识别出该图片是数字:' + str(predint[0]))
