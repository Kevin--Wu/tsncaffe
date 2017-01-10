import caffe 
caffe_root='/home/hadoop/whx/tsncaffe/'
data_root='/home/hadoop/whx/dataset/ucf101/ucf_videoframedata_jpeg/'
model_root='/home/hadoop/whx/exp-result/'
import numpy as np  
import scipy
import sys
import os

net = caffe.Net(caffe_root + 'mywork/ucf101/tsn_bn_inception_rgb_deploy.prototxt',model_root +'2017-1-7/ucf_rgb_bn_inception_iter_80000.caffemodel',caffe.TEST)

shape=(1,9,227,227)
transformer = caffe.io.Transformer({'data': shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

image1=caffe.io.load_image(data_root + 'ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01/frame000001.jpg')
image2=caffe.io.load_image(data_root + 'ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01/frame000010.jpg')
image3=caffe.io.load_image(data_root + 'ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01/frame000020.jpg')
imageinput=np.concatenate((np.concatenate((image1,image2),axis=2),image3),axis=2)
for pix in imageput:
	pix -= 128
print imageinput.shape
os.system("pause")

net.blobs['data'].reshape(1,9,227,227)
net.blobs['data'].data[...] = transformer.preprocess('data', imageinput)
out = net.forward()

print out
