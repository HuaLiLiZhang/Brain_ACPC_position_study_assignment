# -*- coding:utf-8 -*-
"""
Created on 2018/3/29 14:59

@ Author : zhl
"""

###------------------------------------任务--------------------------------------###
#
# 1. 仿照上面的过程，将数据集的标签（label）也输入到TF中，同时保证输入图像和标签一一对应。
#
# 注：对于分类问题，标签可能是一系列数字；对于回归问题，标签可能是输入图像对应的另一幅图像。根据
#    自己要做的方向和具体问题，实现特征和标签成对输入到模型。
#
# 2. 通常在训练时，需要对训练数据进行shuffle处理，但在测试或验证时并不需要，而且为了让输入内容
#    与文件名一一对应，还应该保证输入顺序保持不变。根据上面程序的提示内容，编写一个小程序，实现
#    下面的功能：假设目标文件夹下有100幅图像：001.png, 002.png, ..., 100.png，将这100幅图
#    像依次读入到TF，计算它们像素值的平均值和均方差，然后打印出来。要求打印时按照这样的格式进行：
#         文件名: 平均值/均方差
#    平均值和均方差都只保留2位小数。图像数量自己确定，不能＜5幅图像。
#
#


import tensorflow as tf
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

img_width = 256
img_height = 256
img_slice = 150
channel = 1  #3
batch_size = 3
MAX_VALUE = 32767

# capacity = 5
# max_steps = 1500
# learning_rate = 1e-3
# output_label = np.zeros([1, 4])
# print(output_label.shape)
# # pclabelDir = "E:\\deepLearning\\深度学习\\文献——方向确定\\心脏心室分割\\ACPC\\train_label_acpc\\PC_30.txt"
fileDir = "train_acpc/"
aclabelDir = "train_label_acpc/AC_30.txt"
# logs_train_dir = "E:\\deepLearning\\深度学习\\文献——方向确定\\心脏心室分割\\ACPC\\logs"


def get_file_name (file_dir, ac_dir):
    name_list = []
    for im in os.listdir(file_dir):
        name_list.append(file_dir + im)

    label_ac = []
    f = open(ac_dir)
    line = f.readline()
    while line:
        c, d, e = line.split()
        label_ac.append([c, d, e, 1])
        line = f.readline()
    f.close()
    label_ac = np.array(label_ac)
    file_list = [name_list, label_ac]

    return file_list


def show_slices(train_batch, batch_index, channels, x, y, z):
    # 显示某个batch_size的图像的坐标点，以及三维图像的某一个切面。
    # print(train_label_batch[batch_size-1])
    slices1 = train_batch[batch_index, x, :, :, channels-1] #冠状面
    slices2 = train_batch[batch_index, :, y, :, channels-1] #横断面
    slices3 = train_batch[batch_index, :, :, z, channels-1] #矢状面
    slices = [slices1, slices2, slices3]

    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")
    plt.show()


def get_batch(file_dir, ac_dir, inpW, inpH, inpS, nchannel, batch_size):
    file_list = get_file_name(file_dir, ac_dir)
    num_image = len(file_list)
    if num_image <= 0:
        raise ValueError("Not found Images!")

    # construct ops for reading a single example
    with tf.name_scope('read_single_example'):
        input_img = file_list[0]
        # print(len(file_list[0]))
        label_ac = file_list[1]

    inpBatch = np.zeros([batch_size, inpH, inpW, inpS, nchannel], dtype=np.float32)  # nchannel =1
    # print(inpBatch.shape)
    tarBatch = np.zeros([batch_size, 4])

    # 生成batch，观察图像出现的顺序和文件中图像原来的顺序有什么变化。如果想
    # batch中的图像顺序与图像本身的顺序保持一致，思考应该怎么操作。
    img_index = []
    for i in range(batch_size):
        im = np.random.randint(low=0, high=len(file_list[0]))
        img_index.append(im)
        image = nib.load(input_img[im])
        im_array = image.get_data()
        # print(im_array[:, 130, 75])
        im_array = im_array/MAX_VALUE  #最大值   float64
        # print("/255.0: ", im_array[:, 130, 75])
        im_array = im_array[:, :, :, np.newaxis]
        im_target = label_ac[im]
        tarBatch[i, :] = im_target
        inpBatch[i, :, :, :, :] = im_array

    # inpBatch, tarBatch = tf.train.batch([inpBatch, tarBatch],
    #                                     batch_size = 3,
    #                                     num_threads=4,
    #                                     capacity= 8*batch_size,
    #                                     name='batch_queue')

    with tf.name_scope('cast_to_float64'):
        inpBatch = tf.cast(inpBatch, tf.float64)
        tarBatch = tf.cast(tarBatch, tf.float64)
        img_index = tf.cast(img_index, tf.float64)

    return img_index, inpBatch, tarBatch

img_index, inpBatch, tarBatch = get_batch(fileDir, aclabelDir, img_width, img_height, img_slice, channel, batch_size)

with tf.Session() as sess:
    batch_index = 0
    tf.local_variables_initializer().run()

    inp_Batch, tar_Batch, tarimg_index = sess.run([inpBatch, tarBatch, img_index])
    for i in range(batch_size):
        x = np.random.randint(0, 256)
        y = np.random.randint(0, 256)
        z = np.random.randint(0, 150)
        show_slices(inp_Batch, batch_index, channel, x, y, z)

        print("the batch : ", batch_index,
              " the img_number : ", tarimg_index[i], " the acLabel : ", tar_Batch[i, :])

        # img_mean, img_var = tf.nn.moments(inp_Batch[batch_index, :, :, :, channel-1], [0, 1, 2])
        img_mean = np.mean(inp_Batch[batch_index, :, :, :, channel-1])
        img_var = np.var(inp_Batch[batch_index, :, :, :, channel-1])
        print("the image number : ", tarimg_index[i], " the image_mean is : %.2f " % img_mean,
              " / image_var is : %.6f" % img_var)
        batch_index += 1
        #the batch :  0  the img_number :  2.0  the acLabel :  [ 102.  131.   74.    1.]
        # the image number :  2.0  the image_mean is : 0.01   / image_var is : 0.000243
        # the batch :  1  the img_number :  25.0  the acLabel :  [ 109.  122.   75.    1.]
        # the image number :  25.0  the image_mean is : 0.01   / image_var is : 0.000239
        # the batch :  2  the img_number :  0.0  the acLabel :  [ 123.  137.   74.    1.]
        # the image number :  0.0  the image_mean is : 0.00   / image_var is : 0.000025

sess.close()






