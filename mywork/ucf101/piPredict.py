import caffe 
caffe_root='/home/hadoop/whx/tsncaffe/'
model_root='/home/hadoop/whx/exp-result/'
import numpy as np  
import scipy
import sys
import os
import random

def rgb_video_predict():
	data_root='/home/hadoop/whx/dataset/ucf101/ucf_videoframedata_jpeg/'
	net = caffe.Net(caffe_root + 'mywork/ucf101/pi_bn_inception_rgb_deploy.prototxt',model_root +'model/pi-bn-ucf1-rgb-withpre/pi_bn_rgb_withpre_iter_60000.caffemodel',caffe.TEST)
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
					segnums=[3,6]
					seglength=framenum//6
					i=1
					count=0
					totalout=np.zeros((101))
					while i<seglength:
						frameid = i
						image1path=data_root+curact+'/'+curvideo+'/'+('frame{:0>6d}.jpg'.format(frameid))
						image2path=data_root+curact+'/'+curvideo+'/'+('frame{:0>6d}.jpg'.format(frameid+2*seglength))
						image3path=data_root+curact+'/'+curvideo+'/'+('frame{:0>6d}.jpg'.format(frameid+4*seglength))
						image4path=data_root+curact+'/'+curvideo+'/'+('frame{:0>6d}.jpg'.format(frameid))
						image5path=data_root+curact+'/'+curvideo+'/'+('frame{:0>6d}.jpg'.format(frameid+1*seglength))
						image6path=data_root+curact+'/'+curvideo+'/'+('frame{:0>6d}.jpg'.format(frameid+2*seglength))
						image7path=data_root+curact+'/'+curvideo+'/'+('frame{:0>6d}.jpg'.format(frameid+3*seglength))
						image8path=data_root+curact+'/'+curvideo+'/'+('frame{:0>6d}.jpg'.format(frameid+4*seglength))
						image9path=data_root+curact+'/'+curvideo+'/'+('frame{:0>6d}.jpg'.format(frameid+5*seglength))

						image1=transformer.preprocess('data',caffe.io.load_image(image1path))
						image2=transformer.preprocess('data',caffe.io.load_image(image2path))
						image3=transformer.preprocess('data',caffe.io.load_image(image3path))
						image4=transformer.preprocess('data',caffe.io.load_image(image4path))
						image5=transformer.preprocess('data',caffe.io.load_image(image5path))
						image6=transformer.preprocess('data',caffe.io.load_image(image6path))
						image7=transformer.preprocess('data',caffe.io.load_image(image7path))
						image8=transformer.preprocess('data',caffe.io.load_image(image8path))
						image9=transformer.preprocess('data',caffe.io.load_image(image9path))

						image1[0]-=Bvalue
						image1[1]-=Gvalue
						image1[2]-=Rvalue
						image2[0]-=Bvalue
						image2[1]-=Gvalue
						image2[2]-=Rvalue
						image3[0]-=Bvalue
						image3[1]-=Gvalue
						image3[2]-=Rvalue
						image4[0]-=Bvalue
						image4[1]-=Gvalue
						image4[2]-=Rvalue
						image5[0]-=Bvalue
						image5[1]-=Gvalue
						image5[2]-=Rvalue
						image6[0]-=Bvalue
						image6[1]-=Gvalue
						image6[2]-=Rvalue
						image7[0]-=Bvalue
						image7[1]-=Gvalue
						image7[2]-=Rvalue
						image8[0]-=Bvalue
						image8[1]-=Gvalue
						image8[2]-=Rvalue
						image9[0]-=Bvalue
						image9[1]-=Gvalue
						image9[2]-=Rvalue
						
						net.blobs['data'].reshape(9,3,224,224)
						net.blobs['data'].data[...] = [image1,image2,image3,image4,image5,image6,image7,image8,image9]
						net.forward()

						out = net.blobs['pool_fc'].data[...]
						i+=16
						count+=1
						#totalout=totalout+(out[0][0][0]-totalout)/count
					
						print >> rgbpre, out
						prob=out.argmax()
	                                        print >> rgblabel, (prob,actid)
						print (prob,actid)
						if prob == actid:
							acnum+=1
						totalnum+=1
					print('video %d done' % video_count);
					video_count+=1
		actid+=1
#	rgbpre.write('%d %d\n' % (acnum,totalnum))
	rgbpre.close()
	rgblabel.close()
	return acnum,totalnum

def flow_video_predict():	#The format of flow imgs is flowx flowy flowx flowy
	data_root='/home/hadoop/whx/dataset/ucf101/ucf101_flow_img_tvl1_gpu/'
	net = caffe.Net(caffe_root + 'mywork/ucf101/pi_tsn_bn_inception_flow_deploy.prototxt',model_root +'model/pi-bn-ucf1-flow-withpre/pi_ucf101_split1_tsn_flow_bn_inception_iter_80000.caffemodel',caffe.TEST)
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
					i=1
					count=0
					segnum=6
					flow_length=5
					stopid=(framenum/segnum)-flow_length*2+1
					totalout=np.zeros((101))
					
					while i<stopid:
						curseg=0
						inputId=0
						inputdata=np.zeros((9,10,224,224))
						isreuse=True
						while curseg<segnum:
							frameid = i + curseg*(framenum/segnum)
							j=0
							imagexpath=data_root+curact+'/'+curvideo+'/'+('flow_x_{:0>4d}.jpg'.format(frameid+j))
							imageypath=data_root+curact+'/'+curvideo+'/'+('flow_y_{:0>4d}.jpg'.format(frameid+j))
							imagex=caffe.io.load_image(imagexpath,False)
							imagey=caffe.io.load_image(imageypath,False)
							imagesg=np.concatenate((transformer.preprocess('data',imagex)-mean_value,transformer.preprocess('data',imagey)-mean_value),axis=0)
							j+=1
							while j<flow_length:
								imagexpath=data_root+curact+'/'+curvideo+'/'+('flow_x_{:0>4d}.jpg'.format(frameid+j))
								imageypath=data_root+curact+'/'+curvideo+'/'+('flow_y_{:0>4d}.jpg'.format(frameid+j))
								imagex=caffe.io.load_image(imagexpath,False)
								imagey=caffe.io.load_image(imageypath,False)
								imagesg=np.concatenate((imagesg,np.concatenate((transformer.preprocess('data',imagex)-mean_value,transformer.preprocess('data',imagey)-mean_value),axis=0)),axis=0)
								j+=1

							inputdata[inputId]=imagesg
							inputId+=1
							if(curseg%2==0 and isreuse):
								isreuse=False
								continue
							else:
								isreuse=True
								curseg+=1

						net.blobs['data'].reshape(9,10,224,224)
						net.blobs['data'].data[...] = inputdata
						net.forward()

						out = net.blobs['pool_fc'].data[...]
						i+=32
						count+=1
						#totalout=totalout+(out[0][0][0]-totalout)/count

					
					
					        print >> flowpre, out
					        prob=out.argmax()
    					        flowlabel.write('%d %d\n' % (prob,actid))
                                                print prob,actid
					        if prob == actid:
						    acnum+=1
					        totalnum+=1
                                        print("video %d done" % videonum)
                                        videonum+=1
		actid+=1
#	rgbpre.write('%d %d\n' % (acnum,totalnum))
	flowpre.close()
	flowlabel.close()
	return acnum,totalnum

print flow_video_predict()
