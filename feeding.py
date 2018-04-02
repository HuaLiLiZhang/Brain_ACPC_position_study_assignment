# -*- coding:utf-8 -*-
"""
Created on 2018/3/28 21:10

@ Author : zhl
"""


#
# 1. 仿照上面的过程，使用TF的Feeding机制读入一对LR/HR图像，然后用NN插值，Bilinear和Bicubic
#    插值算法将LR图像放大(到与HR图像一样大)，然后计算插值结果与真实HR图像之间的PSNR和SSIM值。
#
# 注：要求插值过程要在TensorFlow中完成，读入彩色3通道RGB图像，实现两种PSNR和SSIM的计算：
#    1). 在RGB图像的三个通道上同时计算PSNR和SSIM值
#    2). 将RGB图像转化为YCrCb颜色空间，仅在Y通道上计算PSNR和SSIM值。
#    比较这两种方式计算出来的PSNR值和SSIM值有什么差异。
#    from skimage import measure
#    psnr = measure.compare_psnr()
#    ssim = measure.compare_ssim()

#
# 2. 用NumPy实现一个计算总方差的函数，函数输入为NumPy 3D数组，输出为3D数组的TV值。
#
#
# 3. 用你自己写的TV函数，分别计算上述LR图像经过NN，Bilinear和Bicubic插值后的TV值。然后比较
#    插值图像与真实HR图像之间TV值之间的差异。
#
from scipy import misc
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage import measure, color

lr_img_dir = "E:\\deepLearning\\学习任务\\DataInput-Feeding\\DataInput\\dataset\\DIV2K\\DIV2K_train_LR_bicubic\\X4\\"
hr_img_dir = "E:\\deepLearning\\学习任务\\DataInput-Feeding\\DataInput\\dataset\\DIV2K\\DIV2K_train_HR\\"

# 随机从数据集目录下读取一对LR和HR的图像
def read_images(lr_dir, hr_dir):
    lr_names = os.listdir(lr_dir)
    hr_names = os.listdir(hr_dir)
    if(len(lr_names)!=len(hr_names)):
        raise ValueError("LR image size %s does not match HR iamge size %d!" %
                         (len(lr_names), len(hr_names)))
    # 随机生成索引. 由于DIV2K数据集里面的图像都是顺序编号的，所以它们的名称是一一对应的。
    index = np.random.randint(low=0, high=len(lr_names))
    # 读取图片
    lr_image = misc.imread(os.path.join(lr_dir, lr_names[index]))
    hr_image = misc.imread(os.path.join(hr_dir, hr_names[index]))

    return index, lr_image, hr_image


# 根据DIV2K的命名规律，由图像索引/编号生成文件名
def generate_image_name(index):
    im_name = str(index)
    digit = index
    while digit < 1000:
        im_name = "0" + im_name
        digit = digit * 10

    return im_name

def bicubic_image(Lr_image, hr_image):
    # print(hr_image.shape)
    (width, height, channels) = hr_image.shape
    lr_resized_hr = tf.image.resize_images(Lr_image, [width, height], method=0)
# ResizeMethod.BILINEAR: Bilinear interpolation.
# ResizeMethod.NEAREST_NEIGHBOR: Nearest neighbor interpolation.
# ResizeMethod.BICUBIC: Bicubic interpolation.
# ResizeMethod.AREA: Area interpolation.

# img_numpy=img.eval(session=sess)
# print("out2=",type(img_numpy))
# #转化为tensor
# img_tensor= tf.convert_to_tensor(img_numpy)
# print("out2=",type(img_tensor))
#     lr_resized_hr = lr_resized_hr.eval(session=sess)
    #tensor转换为numpy, 也可以，但是不用这么麻烦，后面session绘画的时候，在重新sess.run一下就可以了
    # lr_resized_hr = np.asarray(lr_resized_hr, dtype='float64')
    return lr_resized_hr

def cal_psnr_ssim(lr_bina_hr, hr_image):
    psnr = measure.compare_psnr(hr_image[:, :, 0], lr_bina_hr[:, :, 0])  #只在一个通道上
    ssim = measure.compare_ssim(hr_image[:, :, 0], lr_bina_hr[:, :, 0], multichannel=False)
    return psnr, ssim

#将图像转化为ycrcb的图片格式，在进行计算psnr 和ssim
def rgb_to_ycrcbpsnrssim(lr_bina_hr, hr_image):
    lr_image_y = (color.colorconv.rgb2yuv(lr_bina_hr))[:, :, 0]
    hr_image_y = (color.colorconv.rgb2yuv(hr_image))[:, :, 0]
    psnr = measure.compare_psnr(lr_image_y, hr_image_y)
    ssim = measure.compare_ssim(lr_image_y, hr_image_y, multichannel=False)
    return psnr, ssim

def compute3D_Tv_total_variation(input_3Darray):  #计算三维矩阵的 total_variation 总变分差
    row_val = input_3Darray[1:, :, :]-input_3Darray[:-1, :, :]
    col_val = input_3Darray[:, 1:, :]-input_3Darray[:, :-1, :]
    cha_val = input_3Darray[:, :, 1:] - input_3Darray[:, :, :-1]
    return np.sum(abs(row_val))+np.sum(abs(col_val))+np.sum(abs(cha_val))

#计算低分辨率和高分辨率图像的 total_variation 总变分差
def cal_TV(lr_image, hr_image):
    row_val = lr_image[1:, :, :] - lr_image[:-1, :, :]
    col_val = lr_image[:, 1:, :] - lr_image[:, :-1, :]
    row_val1 = hr_image[1:, :, :] - hr_image[:-1, :, :]
    col_val1 = hr_image[:, 1:, :] - hr_image[:, :-1, :]
    return np.sum(abs(row_val)) + np.sum(abs(col_val)), np.sum(abs(row_val1)) + np.sum(abs(col_val1))

#%%------------------------------------------------------------------------------
# 构建计算图:
# 我们这里的目的是随机读取几张图片，然后分别计算LR和HR图像总变分Total Variation(TV)
#%%------------------------------------------------------------------------------
num_images_to_calc = 1
image_LR = tf.placeholder(dtype=tf.float32, shape=[None, None, 3])
image_HR = tf.placeholder(dtype=tf.float32, shape=[None, None, 3])

# 定义占位符，占位符的具体值在执行sess.run时提供，即所谓的Feeding机制
TV_LR = tf.image.total_variation(image_LR)   #计算TV值。
TV_HR = tf.image.total_variation(image_HR)

with tf.Session() as sess:
    for i in range(num_images_to_calc):
        # 用python代码读取一对图像
        index, lr_image, hr_image = read_images(lr_img_dir, hr_img_dir)
        # 将图像变换成浮点数类型，并归一化。因为placeholder被定义为tf.float32
        # 这里也可以把placeholder定义为tf.uint8, 这样就可以不进行归一化。
        lr_image = lr_image / 255.0
        hr_image = hr_image / 255.0

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 重点在这里：使用TF的Feeding机制计算TV
        lTV, hTV = sess.run([TV_LR, TV_HR],
                            feed_dict={image_LR: lr_image, image_HR: hr_image})
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        print("index: %d" % (index + 1))
        print("LR_TV = %.4f\tHR_TV = %.4f" % (lTV, hTV))  #LR_TV = 43476.9570	HR_TV = 436357.1250

        # 显示图像的索引名字
        im_name = generate_image_name(index + 1)
        #x4线性插值到原图像大小
        lr_binar_hr = bicubic_image(lr_image, hr_image)
        #此时返回为一个tensor的张量，要变为矩阵形式计算psnr和ssim
        lr_binar_hr = sess.run(lr_binar_hr)
        lr_binar_hr = np.asarray(lr_binar_hr, dtype='float64')
        #计算线性插值和原图像的psnr,ssim
        psnr, ssim = cal_psnr_ssim(lr_binar_hr, hr_image)
        #计算RGB转化为YCbCr的psnr ,ssim
        psnr1, ssim1 = rgb_to_ycrcbpsnrssim(lr_binar_hr, hr_image)

        #随机生成三维矩阵，然后计算TV值
        arr = np.random.randint(0, 10, size=[4, 4, 4])
        Tv = compute3D_Tv_total_variation(arr)

        #计算低分辨率和高分辨率的总变分差
        lr_TV, hr_TV = cal_TV(lr_image, hr_image)

        plt.imshow(lr_image)
        plt.title("LR Image %s" % (im_name + "x4.png")), plt.show()
        plt.imshow(hr_image)
        plt.title("HR Image %s" % (im_name + ".png")), plt.show()
        plt.imshow(lr_binar_hr)
        plt.title("lr_binar_to_hr Image %s" % (im_name + ".png")), plt.show()

        print("LR_TV = %.4f\tHR_TV = %.4f" % (lr_TV, hr_TV))  #LR_TV = 43476.9570	HR_TV = 436357.1250
        print("RGB_psnr : ", psnr, "RGB_ssim : ", ssim)
        print("RGB_TO_YcrCb_psnr : ", psnr1, "RGB_To_Ycrcb_ssim", ssim1)
        print("the random 3Darray ", arr, " the TV : ", Tv)





