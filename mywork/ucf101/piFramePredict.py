import caffe 
caffe_root='/home/hadoop/whx/tsncaffe/'
model_root='/home/hadoop/whx/exp-result/'
import numpy as np  
import scipy
import sys
import os
import random

def rgb_frame_predict():
	data_root='/home/hadoop/whx/dataset/ucf101/ucf_videoframedata_jpeg/'
	net = caffe.Net(caffe_root + 'mywork/ucf101/pi_bn_inception_rgb_deploy_frame.prototxt',model_root +'model/pi-bn-ucf1-rgb-withpre/pi_bn_rgb_withpre_iter_60000.caffemodel',caffe.TEST)
	rgbpre=open(caffe_root+'mywork/ucf101/rgbpredict.txt','w')
	rgblabel=open(caffe_root+'mywork/ucf101/rgblabel.txt','w')


	transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
	transformer.set_transpose('data', (2,0,1))
	transformer.set_raw_scale('data', 255)  
	transformer.set_channel_swap('data', (2,1,0))  

	actdirs=os.listdir(data_root)
	actdirs.sort()
	actid=0
	acnum=0
	totalnum=0
	Bvalue=104
	Gvalue=117
	Rvalue=123
	video_count=1
	for curact in actdirs:
		if os.path.isdir(data_root+curact):
			videodirs=os.listdir(data_root+curact)
			videodirs.sort()
			for curvideo in videodirs:
				if os.path.isdir(data_root+curact+'/'+curvideo):
					framelist=os.listdir(data_root+curact+'/'+curvideo)
					framelist.sort()
					framenum = len(framelist)
					frameid=1
					count=0
					totalout=np.zeros((101))
					flow_length=5
					while frameid<(framenum-flow_length+1):
						image1path=data_root+curact+'/'+curvideo+'/'+('frame{:0>6d}.jpg'.format(frameid))
						

						image1=transformer.preprocess('data',caffe.io.load_image(image1path))
						

						image1[0]-=Bvalue
						image1[1]-=Gvalue
						image1[2]-=Rvalue
						
						
						net.blobs['data'].reshape(1,3,224,224)
						net.blobs['data'].data[...] = [image1]
						net.forward()

						out = net.blobs['pool_fc'].data[...]
						#totalout=totalout+(out[0][0][0]-totalout)/count
					
						print >> rgbpre, out
						prob=out.argmax()
						print >> rgblabel, (prob,actid)
						#print (prob,actid)
						if prob == actid:
							acnum+=1
						totalnum+=1

						frameid+=8
					print('video %d done' % video_count);
					video_count+=1
		actid+=1
	print (acnum,totalnum)
	rgbpre.close()
	rgblabel.close()
	return acnum,totalnum

def flow_frame_predict():	#The format of flow imgs is flowx flowy flowx flowy
	data_root='/home/hadoop/whx/dataset/ucf101/ucf101_flow_img_tvl1_gpu/'
	net = caffe.Net(caffe_root + 'mywork/ucf101/pi_tsn_bn_inception_flow_deploy_frame.prototxt',model_root +'../temporal-segment-network/models/ucf101_split_1_tsn_flow_reference_bn_inception.caffemodel',caffe.TEST)
	flowpre=open(caffe_root+'mywork/ucf101/flowpredict.txt','w')
	flowlabel=open(caffe_root+'mywork/ucf101/flowlabel.txt','w')


	transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
	transformer.set_transpose('data', (2,0,1))
	transformer.set_raw_scale('data', 255)  
	mean_value=128

	actdirs=os.listdir(data_root)
	actdirs.sort()
	actid=0
	acnum=0
	totalnum=0
        videonum=1
	for curact in actdirs:
		if os.path.isdir(data_root+curact):
			videodirs=os.listdir(data_root+curact)
			videodirs.sort()
			for curvideo in videodirs:
				if os.path.isdir(data_root+curact+'/'+curvideo):
					framelist=os.listdir(data_root+curact+'/'+curvideo)
					framelist.sort()
					framenum = len(framelist)/2
					frameid=1
					flow_length=5
					totalout=np.zeros((101))
					
					while frameid<=(framenum-flow_length+1):
						inputdata=np.zeros((1,10,224,224))
						imagexpath=data_root+curact+'/'+curvideo+'/'+('flow_x_{:0>4d}.jpg'.format(frameid))
						imageypath=data_root+curact+'/'+curvideo+'/'+('flow_y_{:0>4d}.jpg'.format(frameid))
						imagex=caffe.io.load_image(imagexpath,False)
						imagey=caffe.io.load_image(imageypath,False)
						imagesg=np.concatenate((transformer.preprocess('data',imagex)-mean_value,transformer.preprocess('data',imagey)-mean_value),axis=0)
						j=1
						while j<flow_length:
							imagexpath=data_root+curact+'/'+curvideo+'/'+('flow_x_{:0>4d}.jpg'.format(frameid+j))
							imageypath=data_root+curact+'/'+curvideo+'/'+('flow_y_{:0>4d}.jpg'.format(frameid+j))
							imagex=caffe.io.load_image(imagexpath,False)
							imagey=caffe.io.load_image(imageypath,False)
							imagesg=np.concatenate((imagesg,np.concatenate((transformer.preprocess('data',imagex)-mean_value,transformer.preprocess('data',imagey)-mean_value),axis=0)),axis=0)
							j+=1

						inputdata[0]=imagesg
						

						net.blobs['data'].reshape(1,10,224,224)
						net.blobs['data'].data[...] = inputdata
						net.forward()

						out = net.blobs['pool_fc'].data[...]
					
					        print >> flowpre, out
					        prob=out.argmax()
        					flowlabel.write('%d %d\n' % (prob,actid))
	        				#print prob,actid
		        			if prob == actid:
			        		    acnum+=1
				        	totalnum+=1
                                                
                                                frameid+=8
				print("video %d done" % videonum)
				videonum+=1
		actid+=1
	print (acnum,totalnum)
	flowpre.close()
	flowlabel.close()
	return acnum,totalnum

print flow_frame_predict()
