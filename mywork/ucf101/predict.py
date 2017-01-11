import caffe 
caffe_root='/home/hadoop/whx/tsncaffe/'
model_root='/home/hadoop/whx/exp-result/'
import numpy as np  
import scipy
import sys
import os

def rgb_predict():
	data_root='/home/hadoop/whx/dataset/ucf101/ucf_videoframedata_jpeg/'
	net = caffe.Net(caffe_root + 'mywork/ucf101/tsn_bn_inception_rgb_deploy.prototxt',model_root +'2017-1-7/ucf_rgb_bn_inception_iter_80000.caffemodel',caffe.TEST)

	transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
	transformer.set_transpose('data', (2,0,1))
	transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1))
	transformer.set_raw_scale('data', 255)  
	transformer.set_channel_swap('data', (2,1,0))  

	actdirs=os.listdir(data_root)
	actid=0
	for curact in actdirs:
		if os.path.isdir(data_root+curact):
			videodirs=os.listdir(data_root+curact)
			for curvideo in videodirs:
				if os.path.isdir(data_root+curact+'/'+curvideo):
					framelist=os.listdir(data_root+curact+'/'+curvideo)
					framelist.sort()
					print len(framelist)

	image1=caffe.io.load_image(data_root + 'PlayingViolin/v_PlayingViolin_g01_c01/frame000001.jpg')
	image2=caffe.io.load_image(data_root + 'PlayingViolin/v_PlayingViolin_g01_c01/frame000010.jpg')
	image3=caffe.io.load_image(data_root + 'PlayingViolin/v_PlayingViolin_g01_c01/frame000020.jpg')

	net.blobs['data'].reshape(3,3,224,224)
	net.blobs['data'].data[...] = [transformer.preprocess('data', image1),transformer.preprocess('data', image2),transformer.preprocess('data', image3)]
	net.forward()

	out = net.blobs['pool_fc'].data[...]
	out = out[0][0][0]
	out.argmax()+1
	
	return out	

rgb_predict()
