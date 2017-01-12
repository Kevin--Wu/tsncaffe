import caffe 
caffe_root='/home/hadoop/whx/tsncaffe/'
model_root='/home/hadoop/whx/exp-result/'
import numpy as np  
import scipy
import sys
import os
import random

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
	acnum=0
	totalnum=0
	for curact in actdirs:
		if os.path.isdir(data_root+curact):
			videodirs=os.listdir(data_root+curact)
			for curvideo in videodirs:
				if os.path.isdir(data_root+curact+'/'+curvideo):
					framelist=os.listdir(data_root+curact+'/'+curvideo)
					framelist.sort()
					framenum = len(framelist)
					frameid = random.randint(1,framenum/3)
					image1path=data_root+curact+'/'+curvideo+'/'+('frame{:0>6d}.jpg'.format(frameid))
					image2path=data_root+curact+'/'+curvideo+'/'+('frame{:0>6d}.jpg'.format(frameid+framenum/3))
					image3path=data_root+curact+'/'+curvideo+'/'+('frame{:0>6d}.jpg'.format(frameid+2*framenum/3))

					image1=caffe.io.load_image(image1path)
					image2=caffe.io.load_image(image2path)
					image3=caffe.io.load_image(image3path)

					net.blobs['data'].reshape(3,3,224,224)
					net.blobs['data'].data[...] = [transformer.preprocess('data', image1),transformer.preprocess('data', image2),transformer.preprocess('data', image3)]
					net.forward()

					out = net.blobs['pool_fc'].data[...]
					out = out[0][0][0]
					prob=out.argmax()
					if prob == actid:
						acnum+=1
					totalnum+=1
		actid+=1
	
	return acnum,totalnum	

print rgb_predict()
