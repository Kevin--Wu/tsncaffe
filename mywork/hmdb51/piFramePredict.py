import caffe 
import numpy as np  
import scipy
import sys
import os
import random


caffe_root='/home/hadoop/whx/tsncaffe'
model_root='/home/hadoop/whx/exp-result/model/hmdb51'
szRGBSplitName = "rgb-split1"
szFlowSplitName = "flow-split1"
szWeightfilename = "fusionweight1"
nFlowLength = 5


def fusion_predict():
	rgbdata_root='/home/hadoop/whx/dataset/hmdb51/jpegs_256'
	rgbnet = caffe.Net("{}/{}".format(caffe_root, 'mywork/hmdb51/pi_bn_inception_rgb_frame_deploy.prototxt'), 
		"{}/{}/{}".format(model_root, szRGBSplitName, 'pi_bn_rgb_withpre_iter_100000.caffemodel'), caffe.TEST)
	rgbtransformer = caffe.io.Transformer({'data': rgbnet.blobs['data'].data.shape})
	rgbtransformer.set_transpose('data', (2,0,1))
	rgbtransformer.set_raw_scale('data', 255)  
	rgbtransformer.set_channel_swap('data', (2,1,0))  

	flowdata_root='/home/hadoop/whx/dataset/hmdb51/flowjpg'
	flownet = caffe.Net("{}/{}".format(caffe_root, 'mywork/hmdb51/pi_bn_inception_flow_frame_deploy.prototxt'), 
		"{}/{}/{}".format(model_root, szFlowSplitName, 'pi_bn_flow_withpre_iter_80000.caffemodel'), caffe.TEST)
	flowtransformer = caffe.io.Transformer({'data': flownet.blobs['data'].data.shape})
	flowtransformer.set_transpose('data', (2,0,1))
	flowtransformer.set_raw_scale('data', 255)  



	with open("/home/hadoop/whx/dataset/hmdb51/videotype.txt", "r") as fileVideoType:
		listVideoNameType = fileVideoType.readlines()

	nTotal = 0
	nAcnum = 0
	with open("/home/hadoop/whx/tsncaffe/mywork/hmdb51/{}".format(szWeightfilename),"r") as fusionweight:
		listweight = pickle.load(fusionweight)
	for szLine in listVideoNameType:
		listLine = szLine.split()
		szVideoName = listLine[0]
		nVideoType = int(listLine[1])

		szRgbCurVideoPath = "{}/{}".format(rgbdata_root, szVideoName)
		if not os.path.isdir(szRgbCurVideoPath):
			raise Exception("No such videodir")
		szFlowCurVideoPath = "{}/{}".format(flowdata_root, szVideoName)
		if not os.path.isdir(szFlowCurVideoPath):
			raise Exception("No such videodir")

		listFrames = os.listdir(szRgbCurVideoPath)
		listFrames.sort()
		nRgbFramenum = len(listFrames)

		listFrames = os.listdir(szFlowCurVideoPath)
		listFrames.sort()
		nFlowFramenum = len(listFrames)/2

		if nRgbFramenum != nFlowFramenum:
#			print szVideoName
#			raise Exception("Non-equal framenums between rgb and flow data")
                        nRgbFramenum = nFlowFramenum

		i=1
		while i <= nRgbFramenum - nFlowLength + 1:
			rgbout = rgb_video_predict_commit(i, szRgbCurVideoPath, rgbnet, rgbtransformer)
			flowout = flow_video_predict_commit(i, szFlowCurVideoPath, flownet, flowtransformer)

			szFusionType = "AVEweight"
			if szFusionType == "AVE":
                            out = rgbout + flowout
			elif szFusionType == "MAX":
                            for nId in range(0, len(rgbout[0,0,0])):
                                rgbout[0,0,0,nId] = rgbout[0,0,0,nId] if rgbout[0,0,0,nId] > flowout[0,0,0,nId] else flowout[0,0,0,nId]
                            out = rgbout
                        elif szFusionType == "AVEweight":
                            out = rgbout
                            for nId in range(0, len(rgbout[0,0,0])):
                                out[0,0,0,nId] = rgbout[0,0,0,nId] * listweight[0][nId] + flowout[0,0,0,nId] * listweight[1][nId]
                                
			prob=out.argmax()
			print (prob, nVideoType)
			if prob == nVideoType:
				nAcnum+=1
			nTotal+=1
			i+=2

	print nAcnum, nTotal, nAcnum*1.0/nTotal




def rgb_video_predict():
	data_root='/home/hadoop/whx/dataset/hmdb51/jpegs_256'
	net = caffe.Net("{}/{}".format(caffe_root, 'mywork/hmdb51/pi_bn_inception_rgb_frame_deploy.prototxt'), 
		"{}/{}/{}".format(model_root, szRGBSplitName, 'pi_bn_rgb_withpre_iter_100000.caffemodel'), caffe.TEST)


	transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
	transformer.set_transpose('data', (2,0,1))
	transformer.set_raw_scale('data', 255)  
	transformer.set_channel_swap('data', (2,1,0))  

	with open("/home/hadoop/whx/dataset/hmdb51/videotype.txt", "r") as fileVideoType:
		listVideoNameType = fileVideoType.readlines()

	nTotal = 0
	nAcnum = 0
	for szLine in listVideoNameType:
		listLine = szLine.split()
		szVideoName = listLine[0]
		nVideoType = int(listLine[1])
		
		szCurVideoPath = "{}/{}".format(data_root, szVideoName)
		if not os.path.isdir(szCurVideoPath):
			raise Exception("No such videodir")
		listFrames = os.listdir(szCurVideoPath)
		listFrames.sort()
		nFramenum = len(listFrames)
		i=1
		while i <= nFramenum:
						out = rgb_video_predict_commit(i, szCurVideoPath, net, transformer)
						
						prob=out.argmax()
						print (prob, nVideoType)
						if prob == nVideoType:
							nAcnum+=1
						nTotal+=1
						i+=2

	print nAcnum, nTotal, nAcnum*1.0/nTotal


def rgb_video_predict_commit(i, szCurVideoPath, net, transformer):
						Bvalue=104
						Gvalue=117
						Rvalue=123
						frameid = i
						image1path='{}/frame{:0>6d}.jpg'.format(szCurVideoPath, frameid)

						image1=transformer.preprocess('data',caffe.io.load_image(image1path))

						image1[0]-=Bvalue
						image1[1]-=Gvalue
						image1[2]-=Rvalue
						
						net.blobs['data'].reshape(1,3,224,224)
						net.blobs['data'].data[...] = [image1]
						net.forward()

						out = net.blobs['pool_fc'].data[...]
						return out


def flow_video_predict_commit(i, szCurVideoPath, net, transformer):
						mean_value=128
						inputdata=np.zeros((1,10,224,224))
						isreuse=True
						frameid = i
						j=0
						imagexpath="{}/flow_x_{:0>4d}.jpg".format(szCurVideoPath, frameid+j)
						imageypath="{}/flow_y_{:0>4d}.jpg".format(szCurVideoPath, frameid+j)
						imagex=caffe.io.load_image(imagexpath,False)
						imagey=caffe.io.load_image(imageypath,False)
						imagesg=np.concatenate((transformer.preprocess('data',imagex)-mean_value,transformer.preprocess('data',imagey)-mean_value),axis=0)							
						j+=1	
						while j<nFlowLength:
							imagexpath="{}/flow_x_{:0>4d}.jpg".format(szCurVideoPath, frameid+j)
							imageypath="{}/flow_y_{:0>4d}.jpg".format(szCurVideoPath, frameid+j)
							imagex=caffe.io.load_image(imagexpath,False)
							imagey=caffe.io.load_image(imageypath,False)
							imagesg=np.concatenate((imagesg,np.concatenate((transformer.preprocess('data',imagex)-mean_value,transformer.preprocess('data',imagey)-mean_value),axis=0)),axis=0)
							j+=1

						inputdata[0]=imagesg

						net.blobs['data'].reshape(1,10,224,224)
						net.blobs['data'].data[...] = inputdata
						net.forward()

						out = net.blobs['pool_fc'].data[...]
						return out


def flow_video_predict():	#The format of flow imgs is flowx flowy flowx flowy
	data_root='/home/hadoop/whx/dataset/hmdb51/flowjpg'
	net = caffe.Net("{}/{}".format(caffe_root, 'mywork/hmdb51/pi_bn_inception_flow_frame_deploy.prototxt'), 
		"{}/{}/{}".format(model_root, szFlowSplitName, 'pi_bn_flow_withpre_iter_80000.caffemodel'), caffe.TEST)
#        net = caffe.Net("/home/hadoop/whx/temporal-segment-network/models/hmdb51/tsn_bn_inception_flow_deploy.prototxt",  
#                "/home/hadoop/whx/temporal-segment-network/models/hmdb51_split_1_tsn_flow_reference_bn_inception.caffemodel", caffe.TEST)


	transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
	transformer.set_transpose('data', (2,0,1))
	transformer.set_raw_scale('data', 255)  

	with open("/home/hadoop/whx/dataset/hmdb51/videotype.txt", "r") as fileVideoType:
		listVideoNameType = fileVideoType.readlines()

	nTotal = 0
	nAcnum = 0
	for szLine in listVideoNameType:
		listLine = szLine.split()
		szVideoName = listLine[0]
		nVideoType = int(listLine[1])
		
		szCurVideoPath = "{}/{}".format(data_root, szVideoName)
		if not os.path.isdir(szCurVideoPath):
			raise Exception("No such videodir")
		listFrames = os.listdir(szCurVideoPath)
		listFrames.sort()
		nFramenum = len(listFrames)/2
		i=1
		while i <= nFramenum - nFlowLength + 1:
			out = flow_video_predict_commit(i, szCurVideoPath, net, transformer)

			prob=out.argmax()
			print (prob, nVideoType)
			if prob == nVideoType:
				nAcnum+=1
			nTotal+=1
			i+=16

					
					
	print nAcnum, nTotal, nAcnum*1.0/nTotal



						

#rgb_video_predict()
#flow_video_predict()
fusion_predict()
print "OK"
