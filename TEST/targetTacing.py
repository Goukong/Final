import numpy as np 
import cv2
import math
import os

fway = '/home/talentedyz/文档/Compare/car1'
pway = 'car1_1.txt'

def getPoint(road,line):
	points = open(road)
	ps = ''
	for _ in range(line):
		ps = points.readline()
	#print(ps)
	start = []
	tmp = ''
	for i in ps:
		if i == ',' or i == '\n':
			start.append(int(tmp))
			tmp = ''
		else:
			tmp += i
	return start[0],start[1],start[2],start[3]


def calCenter(points):
	xcenter = points[:,0].sum()/points.shape[0]
	ycenter = points[:,1].sum()/points.shape[0]
	return xcenter,ycenter

def dropBad(find_old,find_new,xcenter,ycenter,T):
	while(1):
		waitForDrop = []
		for i,point in enumerate(find_new):
			xpoint,ypoint = point.ravel()
			distance = pow(xpoint-xcenter,2) + pow(ypoint-ycenter,2)
			if distance > pow(T,2):
				waitForDrop.append(i)
		#没有需要淘汰的特征点时，循环结束
		if len(waitForDrop) == 0:
			break
		#去除不符合条件的特征点
		find_old = np.delete(find_old,waitForDrop,0)
		find_new = np.delete(find_new,waitForDrop,0)
	return find_old,find_new

def calMSE(predict,target,M,N):
	tmp = (target - predict) 
	amount = (tmp*tmp).sum()
	mse = math.sqrt(amount/M*N)
	return mse

def testParam(child,show=False):
	feature_params = dict(
		maxCorners = 1000,
		qualityLevel = child[0],
		minDistance = child[1],
		blockSize = 3
		)
	lk_params = dict(
		winSize = (int(child[2]),int(child[2])),
		maxLevel = int(child[3]),
		criteria = (
			cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
			int(child[4]),child[5])
	)
	mse_list = []
	found_list = []
	flag = 1
	store = os.listdir(fway)
	store.sort()
	#print(store)
	path = fway+'/'+store[flag]
	frame = cv2.imread(path,-1)
	find_new = None
	ix,iy,xLength,yLength = getPoint(pway,flag)
	fx = ix + xLength
	fy = iy + yLength
	radius = min(xLength,yLength)

	getFeature = True

	while frame.any():
		#print(flag)
		if flag > 250:
			break
		#print(path)
		old_gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
		
		if getFeature:	
			mask0 = np.zeros_like(old_gray)
			mask0[iy:fy,ix:fx] = 255
			old_corner = cv2.goodFeaturesToTrack(old_gray,mask=mask0,**feature_params)
			getFeature = False
		else:
			old_corner = find_new.reshape(find_new.shape[0],1,find_new.shape[1])

		flag += 1
		try:
			path = fway+'/'+store[flag]
			frame = cv2.imread(path,-1)
		except:
			print("Finished!")
			break
		new_gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
		new_corner,trace_st,err = cv2.calcOpticalFlowPyrLK(old_gray,new_gray,old_corner,None,**lk_params)

		#get matched points
		find_new = new_corner[trace_st==1]
		find_old = old_corner[trace_st==1]

	

		#calculate center and drop bad points
		xcenter_old,ycenter_old = calCenter(find_new)
		find_old,find_new = dropBad(find_old,find_new,xcenter_old,ycenter_old,radius/2)
		if not find_new.any():
			getFeature = True
			mse_list.append(3000)
			continue

		ix,iy,xLength,yLength = getPoint(pway,flag)
		fx = ix + xLength
		fy = iy + yLength
		radius = min(xLength,yLength)

		centerX = find_new[:,0].sum()/find_new.shape[0]
		centerY = find_new[:,1].sum()/find_new.shape[0]
	
		pix,piy = int(centerX-xLength/2),int(centerY-yLength/2)
		pfx,pfy = int(centerX+xLength/2),int(centerY+yLength/2)

		mask2 = np.zeros_like(frame)
		predict = new_gray[piy:pfy,pix:pfx]
		mask2 = cv2.rectangle(mask2,(pix,piy),(pfx,pfy),(0,0,255),3)
		if show:
			pic= cv2.add(frame,mask2)
			cv2.imshow('pic',pic)
			cv2.waitKey(30)

		if flag % 10 == 0:
			target = new_gray[iy:fy,ix:fx]
			mask3 = np.zeros_like(frame)
			mask3 = cv2.rectangle(mask2,(ix,iy),(fx,fy),(0,255,0),3)
			if show:
				real = cv2.add(frame,mask3)
				cv2.imshow('pic',real)
				cv2.waitKey(30)
			mse = calMSE(predict,target,xLength,yLength)
			#print(mse)
			mse_list.append(mse)
		
	return mse_list

def targetTrace(feature_params,lk_params,base=0):
	flag = 1
	store = os.listdir(fway)
	path = fway+'/'+store[flag+base]
	frame = cv2.imread(path,-1)

	ix,iy,xLength,yLength = getPoint(pway,flag)
	fx = ix + xLength
	fy = iy + yLength
	radius = min(xLength,yLength)

	mse_list = []

	while flag<=10:
		#print(flag)
		old_gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
		
		if flag==1:	
			mask0 = np.zeros_like(old_gray)
			mask0[iy:fy,ix:fx] = 255
			old_corner = cv2.goodFeaturesToTrack(old_gray,mask=mask0,**feature_params)
		else:
			old_corner = find_new.reshape(find_new.shape[0],1,find_new.shape[1])

		flag += 1
		try:
			path = fway+'/'+store[flag+base]
			frame = cv2.imread(path,-1)
		except:
			print("Finished!")
			break
		new_gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
		try:
			new_corner,trace_st,err = cv2.calcOpticalFlowPyrLK(old_gray,new_gray,old_corner,None,**lk_params)	
	
			#get matched points
	
			find_new = new_corner[trace_st==1]
			find_old = old_corner[trace_st==1]
		
		#calculate center and drop bad points
			xcenter_old,ycenter_old = calCenter(find_new)
			find_old,find_new = dropBad(find_old,find_new,xcenter_old,ycenter_old,radius/2)

			ix,iy,xLength,yLength = getPoint(pway,flag)
			fx = ix + xLength
			fy = iy + yLength
			radius = min(xLength,yLength)

		
			if flag % 10 == 0:
				centerX = find_new[:,0].sum()/find_new.shape[0]
				centerY = find_new[:,1].sum()/find_new.shape[0]
		
				pix,piy = int(centerX-xLength/2),int(centerY-yLength/2)
				pfx,pfy = int(centerX+xLength/2),int(centerY+yLength/2)

				predict = new_gray[piy:pfy,pix:pfx]

				target = new_gray[iy:fy,ix:fx]
				mse = calMSE(predict,target,xLength,yLength)
				#print(mse)
		except:
			return -1e4

	return -mse
