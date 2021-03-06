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
szWeightfilename = "fusionweight1"
nFlowLength = 5


def fusion_predict():
	rgbdata_root='/home/hadoop/whx/dataset/hmdb51/jpegs_256'
	rgbnet = caffe.Net("{}/{}".format(caffe_root, 'mywork/hmdb51/pi_bn_inception_rgb_deploy.prototxt'), 
		"{}/{}/{}".format(model_root, szRGBSplitName, 'pi_bn_rgb_withpre_iter_100000.caffemodel'), caffe.TEST)
	rgbtransformer = caffe.io.Transformer({'data': rgbnet.blobs['data'].data.shape})
	rgbtransformer.set_transpose('data', (2,0,1))
	rgbtransformer.set_raw_scale('data', 255)  
	rgbtransformer.set_channel_swap('data', (2,1,0))  

	flowdata_root='/home/hadoop/whx/dataset/hmdb51/flowjpg'
	flownet = caffe.Net("{}/{}".format(caffe_root, 'mywork/hmdb51/pi_bn_inception_flow_deploy.prototxt'), 
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

		nSeglength = nFlowFramenum//6
		i=1
		while i <= nSeglength - nFlowLength + 1:
			rgbout = rgb_video_predict_commit(i, nSeglength, szRgbCurVideoPath, rgbnet, rgbtransformer)
			flowout = flow_video_predict_commit(i, nFlowFramenum, nSeglength, szFlowCurVideoPath, flownet, flowtransformer)

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



def train_fusion_weight(out, rgbout, flowout, listweight):
    nProbId = out.argmax()
    nRgbout = rgbout[0,0,0,nProbId]
    nFlowout = flowout[0,0,0,nProbId]
    
    listweight[0][nProbId] = nRgbout / ( nRgbout + nFlowout )
    listweight[1][nProbId] = nFlowout / ( nRgbout + nFlowout )

    return




def rgb_video_predict():
	data_root='/home/hadoop/whx/dataset/hmdb51/jpegs_256'
	net = caffe.Net("{}/{}".format(caffe_root, 'mywork/hmdb51/pi_bn_inception_rgb_deploy.prototxt'), 
		"{}/{}/{}".format(model_root, szRGBSplitName, 'pi_bn_rgb_withpre_iter_100000.caffemodel'), caffe.TEST)
	rgbpre=open("{}/{}".format(caffe_root, 'mywork/hmdb51/rgbpredict.txt'),'w')
	rgblabel=open("{}/{}".format(caffe_root, 'mywork/hmdb51/rgblabel.txt'),'w')


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
		nSeglength = nFramenum//6
		i=1
		while i <= nSeglength - nFlowLength + 1:
						out = rgb_video_predict_commit(i, nSeglength, szCurVideoPath, net, transformer)
						
					
						print >> rgbpre, out
						prob=out.argmax()
						print (prob, nVideoType)
						if prob == nVideoType:
							nAcnum+=1
						nTotal+=1
						i+=16

	print nAcnum, nTotal, nAcnum*1.0/nTotal
	rgbpre.close()
	rgblabel.close()


def rgb_video_predict_commit(i, nSeglength, szCurVideoPath, net, transformer):
						Bvalue=104
						Gvalue=117
						Rvalue=123
						frameid = i
						image1path='{}/frame{:0>6d}.jpg'.format(szCurVideoPath, frameid)
						image2path='{}/frame{:0>6d}.jpg'.format(szCurVideoPath, frameid+2*nSeglength)
						image3path='{}/frame{:0>6d}.jpg'.format(szCurVideoPath, frameid+4*nSeglength)
						image4path='{}/frame{:0>6d}.jpg'.format(szCurVideoPath, frameid)
						image5path='{}/frame{:0>6d}.jpg'.format(szCurVideoPath, frameid+1*nSeglength)
						image6path='{}/frame{:0>6d}.jpg'.format(szCurVideoPath, frameid+2*nSeglength)
						image7path='{}/frame{:0>6d}.jpg'.format(szCurVideoPath, frameid+3*nSeglength)
						image8path='{}/frame{:0>6d}.jpg'.format(szCurVideoPath, frameid+4*nSeglength)
						image9path='{}/frame{:0>6d}.jpg'.format(szCurVideoPath, frameid+5*nSeglength)

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
						return out


def flow_video_predict_commit(i, nFramenum, nSeglength, szCurVideoPath, net, transformer):
						mean_value=128
						curseg=0
						inputId=0
						inputdata=np.zeros((9,10,224,224))
						isreuse=True
						segnum = 6
						while curseg<segnum:
							frameid = i + curseg*(nFramenum/segnum)
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
						return out


def flow_video_predict():	#The format of flow imgs is flowx flowy flowx flowy
	data_root='/home/hadoop/whx/dataset/hmdb51/flowjpg'
	net = caffe.Net("{}/{}".format(caffe_root, 'mywork/hmdb51/pi_bn_inception_flow_deploy.prototxt'), 
		"{}/{}/{}".format(model_root, szFlowSplitName, 'pi_bn_flow_withpre_iter_80000.caffemodel'), caffe.TEST)
	flowpre=open("{}/{}".format(caffe_root, 'mywork/hmdb51/flowpredict.txt'),'w')


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
		nSeglength = nFramenum//6
		i=1
		while i <= nSeglength - nFlowLength + 1:
			out = flow_video_predict_commit(i, nFramenum, nSeglength, szCurVideoPath, net, transformer)
			print >> flowpre, out
			prob=out.argmax()
                        print (prob, nVideoType)
			if prob == nVideoType:
				nAcnum+=1
			nTotal+=1
			i+=16

					
					
	print nAcnum, nTotal, nAcnum*1.0/nTotal
	flowpre.close()



						

#rgb_video_predict()
#flow_video_predict()
fusion_predict()
print "OK"
