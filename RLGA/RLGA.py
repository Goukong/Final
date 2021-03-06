import numpy as np 
import cv2
import targetTracing as TT
import RL_brain
import matplotlib.pyplot as plt 
class RLGA:
	"""docstring for RLGA"""
	def __init__(self,DNA_SIZE = 6,POP_SIZE = 100):
		self.dna_size = DNA_SIZE
		self.pop_size = POP_SIZE
		self.min_mse = -1e10
		self.base = 0
		self.max_reward = -9999
		self.max_param = []

	def changeTrainSet(self):
		self.base += 10
		#self.min_mse = -1e10
		self.max_found = -1
		self.max_param = []
		self.max_reward = -9999
		self.mseList = []
		self.foundList = []

	def createQTable(self):
		#First we devide the dna into 6 parts
		#and create 6 q-table with 100 action choice 
		t1 = RL_brain.QLearningTable(self.pop_size)
		t2 = RL_brain.QLearningTable(self.pop_size)
		t3 = RL_brain.QLearningTable(self.pop_size)
		t4 = RL_brain.QLearningTable(self.pop_size)
		t5 = RL_brain.QLearningTable(self.pop_size)
		t6 = RL_brain.QLearningTable(self.pop_size)
		agents =[t1,t2,t3,t4,t5,t6]

		return agents

	def createPopulation(self):
		#create a module for DNA
		dna_mod = np.empty((self.pop_size,self.dna_size))
		#each DNA has its own bound
		for i in range(self.pop_size):
			#qualityLevel [0.01,0.1]
			dna_mod[i][0] = np.random.rand()/10
		
			#mindistance  [0,100] 
			dna_mod[i][1] = np.random.randint(0,100,1)
		
			#winSize  (2,100]
			dna_mod[i][2] = np.random.randint(3,100,1)
		
			#maxlevel,COUNT [0,100]
			dna_mod[i][3] = np.random.randint(0,100,1)
			dna_mod[i][4] = np.random.randint(0,100,1)
		
			#EPS [0,1]
			dna_mod[i][5] = np.random.rand()
		self.pop = dna_mod
		return dna_mod

	def getfitness(self,child):
		#translate the DNA into params
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
		#!!!!get MSE
		mse = TT.targetTrace(feature_params,lk_params,self.base) 
		return mse

	def readPastExperience(self,agents):
		self.pop = np.fromfile('pop.bin',dtype = np.float64)
		self.pop = self.pop.reshape(self.pop_size,self.dna_size)
		min_mse = np.fromfile('min_mse.bin',dtype=np.float64)
		self.min_mse = float(min_mse)
		for i,agent in enumerate(agents):
			agent.readPastInfor(i)
							
		'''
			读取种群信息
			读取Q表信息
		'''
	def writeCurrentInfor(self,agents):
		self.pop.tofile("pop.bin")
		for i,agent in enumerate(agents):
			agent.writeInfor(i)
		min_mse = np.array(self.min_mse)
		min_mse.tofile('min_mse.bin')

	def getAChild(self,agents):
		child = []
		for i in range(5):
			list = []
			child.append(list)
		for i,agent in enumerate(agents):
			idx = agent.choose_best()
			for m,num in enumerate(idx):
				num = int(num)	
				parent = self.pop[num]
				child[m].append(parent[i])
		return child


	#Simulate and update GA 
	def RL(self,agents):
		#create mods to store dna and choosed pop
		child = []
		state = []
		for i,agent in enumerate(agents):
			idx = agent.choose_action()
			state.append(idx)
			#print(self.pop.shape)
			idx = int(idx)
			parent = self.pop[idx]
			child.append(parent[i])

		#get reward
		'''
			得到适应度，并且拿到最适合的参数
			计算出奖励值
		'''
		mse = self.getfitness(child)
		reward = 0

		if self.min_mse == -1e10:
			self.min_mse = mse
			self.last_mse = mse
			mse_reward = 0
		else:
			mse_reward = mse - self.min_mse
			if (mse_reward > 0) and (mse != 0):
				self.min_mse = mse
		#print(self.min_mse)
		if mse == 0:
			reward = 0
		else:
			reward = mse_reward

		if reward > self.max_reward:
			self.max_reward = reward
			self.max_param = child

		#train the q_table
		for i,agent in enumerate(agents):
			agent.learn(state[i],reward)

		return agents
		