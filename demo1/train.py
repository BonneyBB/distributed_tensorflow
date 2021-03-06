import tensorflow as tf
import time
from tensorflow.examples.tutorials.mnist import input_data

# 载入数据
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# 每个批次的大小
batch_size = 100
# 计算一共有多少个批次
n_batch = mnist.train.num_examples


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
        W_fc2 = weight_variable([1024, 10])  # 上一层有1024个神经元，全连接层有10个神经元
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

# 合并所有的summary
saver = tf.train.Saver()
init_op = tf.global_variables_initializer()
summary_op = tf.summary.merge_all()


with tf.Session() as sess:
    sess.run(init_op)
    train_writer = tf.summary.FileWriter('./logs', sess.graph)
    start_time = time.time()
    for step in range(10000):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        summary, _ = sess.run([summary_op, train_step],
                              feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.7})
        train_writer.add_summary(summary, step)
        if step % 100 == 0:
            test_acc = sess.run(accuracy,
                                feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
            train_acc = sess.run(accuracy,
                                 feed_dict={x: mnist.train.images, y: mnist.train.labels, keep_prob: 1.0})
            print("Iter " + str(step) + ", test_acc: " + str(test_acc) + ", train_acc = " + str(train_acc))
    print('Training time: %3.2fs' % (time.time() - start_time))
    saver.save(sess, 'model/train.ckpt')



