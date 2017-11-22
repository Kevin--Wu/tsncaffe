import caffe 
import numpy as np  
import scipy
import sys
import os
import random
import pickle


caffe_root='/home/hadoop/whx/tsncaffe'
model_root='/home/hadoop/whx/exp-result/model/hmdb51'
szRGBSplitName = "rgb-split1"
szFlowSplitName = "flow-split1"
szTrainsplit = "TrainSplit1"
szWeightfilename = "fusionweight1"
nFlowLength = 5
nEpochsize = 1


def record_fusion():
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



	with open("/home/hadoop/whx/dataset/hmdb51/{}.txt".format(szTrainsplit), "r") as fileVideoType:
		listVideoNameType = fileVideoType.readlines() # You need to REPLACE the flowjpg with rgb(jepgs_256)

	nTotal = 0
	nAcnum = 0
	# listweight = [ [ 0.5 for listYid in range(51) ] for listXid in range(2) ]

	nepochnum = 1
	listOutput = None
	listLabel = list()

	while nepochnum <= nEpochsize:
		for szLine in listVideoNameType:
			listLine = szLine.split()
			szVideoName = listLine[0]
			nVideoType = int(listLine[2])

			szFlowCurVideoPath = "{}".format(szVideoName)
			if not os.path.isdir(szFlowCurVideoPath):
				raise Exception("No such videodir")
			szRgbCurVideoPath = "{}".format(szVideoName.replace('flowjpg','jpegs_256'))
			if not os.path.isdir(szRgbCurVideoPath):
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
				# rgbout = rgb_video_predict_commit(i, szRgbCurVideoPath, rgbnet, rgbtransformer)
				# flowout = flow_video_predict_commit(i, szFlowCurVideoPath, flownet, flowtransformer)
				# out = rgbout
				# for nId in range(0, len(rgbout[0,0,0])):
				# 	out[0,0,0,nId] = rgbout[0,0,0,nId] * listweight[0][nId] + flowout[0,0,0,nId] * listweight[1][nId]

				# out = rgbout + flowout
				# prob=out.argmax()
				# print (prob, nVideoType)
				# if prob == nVideoType:
				# 	train_fusion_weight(out, rgbout, flowout, listweight)
				# 	nAcnum+=1
				# nTotal+=1
				# if listOutput is not None:
				# 	listOutput = np.concatenate((listOutput, rgbout), axis = 0)
				# 	listOutput = np.concatenate((listOutput, flowout), axis = 0)
				# else:
				# 	listOutput = rgbout
				# 	listOutput = np.concatenate((listOutput, flowout), axis = 0)
				listLabel.append(nVideoType)
                                

				i+=2

		nepochnum += 1
		print nepochnum

	with open("/home/hadoop/whx/tsncaffe/mywork/hmdb51/fusionlabel","w") as fusionlabel:
		pickle.dump(listLabel, fusionlabel)
	# with open("/home/hadoop/whx/tsncaffe/mywork/hmdb51/fusionoutput","w") as fusionoutput:
	# 	pickle.dump(listOutput, fusionoutput)

	#print nAcnum, nTotal, nAcnum*1.0/nTotal
	return


def train_fusion():
	with open("/home/hadoop/whx/tsncaffe/mywork/hmdb51/fusionoutput","r") as fusionoutput:
		listOutput = pickle.load(fusionoutput)
	with open("/home/hadoop/whx/tsncaffe/mywork/hmdb51/fusionlabel","r") as fusionlabel:
		listLabel = pickle.load(fusionlabel)

	arrayShape = listOutput.shape
	nVideoNum = len(listLabel)
	nId = 0
	listweight = [ [ 0.5 for listYid in range(51) ] for listXid in range(2) ]
    while nId < nVideoNum:
    	nVideoType = listLabel[nId]
    	rgbout = listOutput[2*nId - 1]
    	flowout = listOutput[2*nId]

    	out = rgbout
		for nOutid in range(0, len(rgbout[0,0])):
			out[0,0,nOutid] = rgbout[0,0,nOutid] * listweight[0][nOutid] + flowout[0,0,nOutid] * listweight[1][nOutid]
    	prob=out.argmax()
		print (prob, nVideoType)
		if prob == nVideoType:
			train_fusion_weight(out, rgbout, flowout, listweight)

		nId += 1



def train_fusion_weight(out, rgbout, flowout, listweight):
    nProbId = out.argmax()
    nRgbout = rgbout[0,0,nProbId]
    nFlowout = flowout[0,0,nProbId]
    
    listweight[0][nProbId] = nRgbout / ( nRgbout + nFlowout )
    listweight[1][nProbId] = nFlowout / ( nRgbout + nFlowout )

    return



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




						

#rgb_video_predict()
#flow_video_predict()
#record_fusion()
train_fusion()
print "OK"
