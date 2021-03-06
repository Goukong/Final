import RLGA
import matplotlib.pyplot as plt
import numpy as np 
import targetTacing
import datetime
def getMaxParam():
	rlga = RLGA.RLGA()
	agents = rlga.createQTable()
	pop = rlga.createPopulation()
	firstTime = True
	max_reward = 0
	max_param = []
	if not firstTime:
		rlga.readPastExperience(agents)
		#print(rlga.pop)
	for i in range(5):
		for _ in range(20):
			agents = rlga.RL(agents)
		#x = np.linspace(0,10,50)
		#y1 = rlga.mseList
		#y2 = rlga.foundList
		#plt.plot(x,y1)
		#plt.plot(x,y2)
		#plt.show()
		print('max_reward:',rlga.max_reward)
		print('max_param:',rlga.min_param)
		if rlga.max_reward > max_reward:
			max_reward = rlga.max_reward
			max_param = rlga.min_param
		rlga.changeTrainSet()
	rlga.writeCurrentInfor(agents)	
	result = rlga.getAChild(agents)
	#print(result)
	return result,max_param


if __name__ == '__main__':
	np.seterr(divide='ignore',invalid='ignore')
	starttime = datetime.datetime.now()
	r,m = getMaxParam()
	endtime = datetime.datetime.now()
	print('findparam_time:',(endtime - starttime).seconds)

	print('final:',r)
	min_mse = 1e5
	min_param = []
	amount = 0
	for param in r:
		mse_list = targetTacing.testParam(param)
		average = sum(mse_list)/len(mse_list)
		amount += average
		if average<min_mse:
			min_mse = average
			min_param = param

	endtime = datetime.datetime.now()
	print('finished_time:',(endtime - starttime).seconds)

	print('min_average_mse:',min_mse)
	print('min_param',min_param)
	print('amount',amount/5)
	mse_list = targetTacing.testParam(min_param,show=True)
	length = len(mse_list)
	x = range(length)
	plt.plot(x,mse_list)
	plt.show()
	