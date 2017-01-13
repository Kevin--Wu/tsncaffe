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
	net = caffe.Net(caffe_root + 'mywork/ucf101/tsn_bn_inception_rgb_deploy.prototxt',model_root +'model/tsp-bn-ucf1-rgb/ucf_rgb_bn_inception_iter_80000.caffemodel',caffe.TEST)
	rgbpre=open(caffe_root+'mywork/ucf101/rgbpredict.txt','w')
	rgblabel=open(caffe_root+'mywork/ucf101/rgblabel.txt','w')


	transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
	transformer.set_transpose('data', (2,0,1))
	transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1))
	transformer.set_raw_scale('data', 255)  
	transformer.set_channel_swap('data', (2,1,0))  

	actdirs=os.listdir(data_root)
	actdirs.sort()
	actid=0
	acnum=0
	totalnum=0
	for curact in actdirs:
		if os.path.isdir(data_root+curact):
			videodirs=os.listdir(data_root+curact)
			videodirs.sort()
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
					print >> rgbpre, out

					prob=out.argmax()
					rgblabel.write('%d %d\n' % (prob,actid))
					if prob == actid:
						acnum+=1
					totalnum+=1
		actid+=1
#	rgbpre.write('%d %d\n' % (acnum,totalnum))
	rgbpre.close()
	rgblabel.close()
	return acnum,totalnum

def flow_predict() #光流输入的格式是xyxyxy...
	data_root='/home/hadoop/whx/dataset/ucf101/ucf101_flow_img_tvl1_gpu/'
	net = caffe.Net(caffe_root + 'mywork/ucf101/tsn_bn_inception_flow_deploy.prototxt',model_root +'model/tsp-bn-ucf1-flow-withpre/ucf101_split1_tsn_flow_bn_inception_iter_80000.caffemodel',caffe.TEST)
	flowpre=open(caffe_root+'mywork/ucf101/flowpredict.txt','w')
	flowlabel=open(caffe_root+'mywork/ucf101/flowlabel.txt','w')


	transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
	transformer.set_transpose('data', (2,0,1))
	transformer.set_raw_scale('data', 255)  

	actdirs=os.listdir(data_root)
	actdirs.sort()
	actid=0
	acnum=0
	totalnum=0
	for curact in actdirs:
		if os.path.isdir(data_root+curact):
			videodirs=os.listdir(data_root+curact)
			videodirs.sort()
			for curvideo in videodirs:
				if os.path.isdir(data_root+curact+'/'+curvideo):
					framelist=os.listdir(data_root+curact+'/'+curvideo)
					framelist.sort()
					framenum = len(framelist)
					i=1
					count=0
					segnum=3
					flow_length=5
					stopid=(framenum/3)-flowlength*2+1
					totalout=[0]*101
					
					while i<stopid:
						curseg=0
						inputdata=[]
						while curseg<segnum:
							frameid = i + curseg*(framenum/3)
							j=0
							imagexpath=data_root+curact+'/'+curvideo+'/'+('flow_x_{:0>4d}.jpg'.format(frameid+j))
							imageypath=data_root+curact+'/'+curvideo+'/'+('flow_y_{:0>4d}.jpg'.format(frameid+j))
							imagex=caffe.io.load_image(imagexpath)
							imagey=caffe.io.load_image(imageypath)
							imagesg=concatenate((transformer.preprocess('data',imagex),transformer.preprocess('data',imagey)),axis=0)
							j+=1
							while j<flow_length:
								imagexpath=data_root+curact+'/'+curvideo+'/'+('flow_x_{:0>4d}.jpg'.format(frameid+j))
								imageypath=data_root+curact+'/'+curvideo+'/'+('flow_y_{:0>4d}.jpg'.format(frameid+j))
								imagex=caffe.io.load_image(imagexpath)
								imagey=caffe.io.load_image(imageypath)
								imagesg=concatenate((imagesg,concatenate((transformer.preprocess('data',imagex),transformer.preprocess('data',imagey)),axis=0)),axis=0)
								j+=1

							inputdata.append(imagesg)

						net.blobs['data'].reshape(3,10,224,224)
						net.blobs['data'].data[...] = inputdata
						net.forward()

						out = net.blobs['pool_fc'].data[...]
						i+=2
						count+=1
						totalout=totalout+(out[0][0][0]-totalout)/count

					
					
					print >> flowpre, totalout
					prob=out.argmax()
					flowlabel.write('%d %d\n' % (prob,actid))
					if prob == actid:
						acnum+=1
					totalnum+=1
		actid+=1
#	rgbpre.write('%d %d\n' % (acnum,totalnum))
	flowpre.close()
	flowlabel.close()
	return acnum,totalnum

print flow_predict()
