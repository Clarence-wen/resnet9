import os
import tensorflow as tf
from network import resnet9
import time
import numpy as np
import cv2
import pickle
import random

data_path = "./cifar-10"
number_class = 10
image_size = [32,32]
batch_size = 128
learn_rate = 0.001
num_epochs = 100000
save_step = 10000
output_path = "./model"

if not os.path.exists(output_path):
    os.makedirs(output_path)

cifar10_mean = (0.4914, 0.4822, 0.4465) # equals np.mean(train_set.train_data, axis=(0,1,2))/255
cifar10_std = (0.2471, 0.2435, 0.2616) # equals np.std(train_set.train_data, axis=(0,1,2))/255

def normalise(x, mean=cifar10_mean, std=cifar10_std):
    x, mean, std = [np.array(a, np.float32) for a in (x, mean, std)]
    x -= mean*255
    x *= 1.0/(255*std)
    return x


def build_network():
    X = tf.placeholder(tf.float32, shape=[None, image_size[0],image_size[1],3])
    Y = tf.placeholder(tf.float64, shape=[None, number_class])
    pre_Y = resnet9(X,number_class)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pre_Y,labels=Y))
    tf.summary.scalar("loss",cost)
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(learning_rate=learn_rate,
                                               global_step=global_step,
                                               decay_steps=5000,
                                               decay_rate=0.85,
                                               staircase=True)
    tf.summary.scalar('learning_rate', learning_rate)

    optimize = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(cost)

    summary_op = tf.summary.merge_all()

    return dict(x=X, y=Y, optimize=optimize, cost=cost, summary=summary_op, global_step=global_step)

def load_data(is_train=True,shuffle=True):

    label_bytes = 1
    image_bytes = image_size[0] * image_size[1] * 3

    with tf.name_scope('input'):

        if is_train:
            filenames = [os.path.join(data_path, 'data_batch_%d.bin' % ii)
                         for ii in range(1, 6)]
        else:
            filenames = [os.path.join(data_path, 'test_batch.bin')]
        #异常处理
        for f in filenames:
            if not gfile.Exists(f):
                raise ValueError('Failed to find file: ' + f)

        filename_queue = tf.train.string_input_producer(filenames)

        reader = tf.FixedLengthRecordReader(label_bytes + image_bytes)

        key, value = reader.read(filename_queue)

        record_bytes = tf.decode_raw(value, tf.uint8)

        label = tf.cast(tf.slice(record_bytes, [0], [label_bytes]), tf.int32)

        image_raw = tf.reshape(tf.slice(record_bytes, [label_bytes], [image_bytes]), [3, image_size[0], image_size[1]])

        image = tf.transpose(image_raw, [1, 2, 0])  # convert from D/H/W to H/W/D
        image = tf.cast(image, tf.float32)

        #随机裁剪
        distorted_image = tf.random_crop(image, [image_size[0], image_size[1], 3])

        # 随机水平翻转图像
        distorted_image = tf.image.random_flip_left_right(distorted_image)

        # 由于这些操作是不可交换的，因此可以考虑随机化和调整操作的顺序
        # 在某范围随机调整图片亮度
        distorted_image = tf.image.random_brightness(distorted_image,
                                                     max_delta=63)
        # 在某范围随机调整图片对比度
        distorted_image = tf.image.random_contrast(distorted_image,
                                                   lower=0.2, upper=1.8)

        # 减去平均值并除以像素的方差，白化操作：均值变为0，方差变为1
        image = tf.image.per_image_standardization(distorted_image)

        # 设置张量的形状.
        # image.set_shape([image_size, image_size, 3])

        # label.set_shape([1])

        if shuffle:
            images, label_batch = tf.train.shuffle_batch(
                [image, label],
                batch_size=batch_size,
                num_threads=64,
                capacity=20000,
                min_after_dequeue=3000)
        else:
            images, label_batch = tf.train.batch(
                [image, label],
                batch_size=batch_size,
                num_threads=64,
                capacity=2000)
        # ONE-HOT
        n_classes = 10

        label_batch = tf.one_hot(label_batch, depth=n_classes)
        label_batch = tf.cast(label_batch, dtype=tf.int32)
        label_batch = tf.reshape(label_batch, [batch_size, n_classes])

        return images, label_batch
'''
def load_data(training = True):
    train_data =[]
    label_data =[]
    if training:
        for i in range(5):
            with open("/home/clarence/classification_program/cifar10-fast-tf/cifar-10/data_batch_" + str(i + 1),
                      mode='rb') as file:
                data = pickle.load(file, encoding='bytes')
                train_data+= list(data[b'data'])
                label_data += data[b'labels']
    else:
        with open("/home/clarence/classification_program/cifar10-fast-tf/cifar-10/test_batch",mode='rb') as file:
            data = pickle.load(file, encoding='bytes')
            train_data += list(data[b'data'])
            label_data += data[b'labels']

    train_data = np.array(train_data)
    # print(np.shape(train_data),np.shape(train_data)[0])
    train_data = np.reshape(train_data,[np.shape(train_data)[0],3,image_size[0],image_size[1]])
    train_data = np.transpose(train_data, [0, 2, 3, 1])

    label_data = np.array(label_data)
    # print(np.shape(label_data), np.shape(label_data)[0])
    label_data = np.reshape(label_data, [np.shape(label_data)[0], 1])

    random_number = [i for i in range(len(train_data))]
    random.shuffle(random_number)
    # print(random_number[:batch_size],type(train_data),np.shape(train_data))
    train_data_batch = train_data[random_number[:batch_size],:,:,:]
    label_data_batch = label_data[random_number[:batch_size],:]

    #one_hot 编码
    label_data_batch = one_hot_encode(label_data_batch)

    return train_data_batch, label_data_batch
'''
def one_hot_encode(label_data):
    zeros_class = np.zeros((np.shape(label_data)[0],number_class))
    for i in range(np.shape(label_data)[0]):
        zeros_class[i][label_data[i]]=1
    return zeros_class

def train():

    graph = build_network()
    saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    summary_writer = tf.summary.FileWriter("./log",sess.graph)
    sess.run(tf.global_variables_initializer())

    checkpoint = tf.train.get_checkpoint_state(output_path)
    if (checkpoint != None):
        tf.logging.info("Restoring full model from checkpoint file %s", checkpoint.model_checkpoint_path)
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("---load model --%s" % checkpoint.model_checkpoint_path)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    for  step in range(1,num_epochs):
        old_time =time.time()

        img_batch_train, label_batch_train = load_data()
        # print(np.shape(img_batch_train))
        # cv2.imwrite("./1.jpg",img_batch_train[0])
        # print(label_batch_train)
        #img_batch_train = normalise(img_batch_train, mean=cifar10_mean, std=cifar10_std)
        # print(type(img_batch_train),np.shape(label_batch_train))
        _,training_loss,summary= sess.run([graph["optimize"],graph["cost"],graph["summary"]],
                                                        feed_dict={graph["x"]:np.float32(img_batch_train),graph["y"]:np.float32(label_batch_train),graph["global_step"]: step})
        # print(label_batch_train)
        now_time =time.time()
        if step%50 ==0 or step+1 ==num_epochs:
            print("the step is:  %d  --- training loss is:  %f --- time is: %f  "%(step,training_loss,now_time-old_time))

            summary_writer.add_summary(summary,step)
        if step%100 ==0 or step+1 ==num_epochs:
            val_image, val_label = load_data(training=False)

            val_image = normalise(val_image, mean=cifar10_mean, std=cifar10_std)
            val_loss = sess.run(graph["cost"],
                                feed_dict={graph["x"]: np.float32(val_image),
                                           graph["y"]: np.float32(val_label), graph["global_step"]: step})
            print(type(val_loss))
            print("###################the step is:  %d  --- val loss is:  %f --- time is: %f  " % (step, val_loss, now_time - old_time))
            summary_writer.add_summary(summary, step)

        if (step-1)%save_step==0:
            saver.save(sess, output_path+"/Vgg16_%d.ckpt"%(step-1))
    coord.request_stop()
    coord.join(threads)
    sess.close()

if __name__=="__main__":
    train()























