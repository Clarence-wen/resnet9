import os
import tensorflow as tf
from network import resnet9
import time
import numpy as np
from tensorflow.python.platform import gfile
import cv2


data_path = "./cifar-10"
num_classes = 58
image_size = [32,32]
batch_size = 32
learn_rate = 0.0001
num_epochs = 100000
save_step = 10000
output_path = "./model"


if not os.path.exists(output_path):
    os.makedirs(output_path)

VGG_MEAN_rgb = [123.68, 116.779, 103.939]
def reader():
    print("++++++++++++++++++++")
    all_images_path = np.genfromtxt('/media/clarence/F1/雪贝尔/image.txt', delimiter='\t', dtype=np.str)[1:]
    print("-------------------")
    all_labels = np.genfromtxt('/media/clarence/F1/雪贝尔/classes_to_image.txt', delimiter='\t', dtype=np.int32)[1:]
    file_dir_queue = tf.train.slice_input_producer([all_images_path, all_labels], shuffle=True, capacity=512)
    img_contents = tf.read_file(file_dir_queue[0])
    label = tf.cast(tf.one_hot(file_dir_queue[1], num_classes), tf.float32)
    image = tf.image.decode_jpeg(img_contents, channels=3)

    image = tf.cast(image, tf.float32)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.3)

    image = image - VGG_MEAN_rgb
    image = tf.image.resize_image_with_crop_or_pad(image, image_size[0], image_size[1])

    image_batch, label_batch = tf.train.batch([image, label], batch_size, capacity=128)
    return image_batch, label_batch


def build_network():
    X = tf.placeholder(tf.float32, shape=[None, image_size[0],image_size[1],3])
    Y = tf.placeholder(tf.float64, shape=[None, num_classes])
    pre_Y = resnet9(X,num_classes)

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

def train():
    img_batch, label_batch = reader()
    # val_img_batch, val_label_batch= load_data(is_train=False,shuffle=False)
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
        img_batch_train,label_batch_train  =sess.run([img_batch,label_batch])
        # cv2.imwrite("./1.jpg",img_batch_train[0])
        # print(label_batch_train)
        # img_batch_train = normalise(img_batch_train, mean=cifar10_mean, std=cifar10_std)
        # print(type(img_batch_train),np.shape(label_batch_train))
        _,training_loss,summary= sess.run([graph["optimize"],graph["cost"],graph["summary"]],
                                                        feed_dict={graph["x"]:np.float32(img_batch_train),graph["y"]:np.float32(label_batch_train),graph["global_step"]: step})
        # print(label_batch_train)
        now_time =time.time()

        print("the step is:  %d  --- training loss is:  %f --- time is: %f  "%(step,training_loss,now_time-old_time))

        summary_writer.add_summary(summary,step)
        if (step-1)%save_step==0:
            saver.save(sess, output_path+"/resnet9_%d.ckpt"%(step-1))
    coord.request_stop()
    coord.join(threads)
    sess.close()

if __name__=="__main__":
    train()























