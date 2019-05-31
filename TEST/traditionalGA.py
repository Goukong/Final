#将GA在自适应提取上复现
#然后将select功能用RL方式复现

import numpy as np
import cv2
import targetTacing as TT 
import matplotlib.pyplot as plt
import datetime

DNA_SIZE = 6
POP_SIZE = 100
N_GENERATION = 20
CROSS_RATE = 0.8 
MUTATION_RATE = 0.003

DNA_BOUND1 = [0.01,0.1] # for qualityLevel
DNA_BOUND2 = [0,100] # for minDistance,winSize,maxlevel,COUNT
DNA_BOUND3 = [0,1] #for EPS

def get_fitness(pop):
	#store results from targetTracing
	fitness = np.empty(POP_SIZE)
	#get the fitness
	for i in range(POP_SIZE):
		#translate the DNA into params
		feature_params = dict(maxCorners = 1000,
			qualityLevel = pop[i][0],
			minDistance = pop[i][1],
			blockSize = 3
			)
		lk_params = dict(winSize = (int(pop[i][2]),int(pop[i][2])),
			maxLevel = int(pop[i][3]),
			criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,int(pop[i][4]),pop[i][5])
			)
		#get the tracing result from tragetTrace func as fitness
		mse = TT.targetTrace(feature_params,lk_params)
		#print(mse)
		fitness[i] = mse
	return fitness

def select(pop,fitness):
	#Survival of the fittest 
	#the param is True,so each time the selection is from the whole size
	idx = np.random.choice(POP_SIZE,POP_SIZE,True,fitness/fitness.sum())
	return pop[idx]

def crossover(parent,pop):
	if np.random.rand() < CROSS_RATE:
		#random choose one pop to cross with the current 
		i_ = np.random.randint(0,POP_SIZE,1)
		choice =np.random.randint(0,2,size = DNA_SIZE).astype(np.bool)
		mom = pop[i_].flatten()
		parent[choice] = mom[choice]
	return parent 

def mutate(child):
	#even if the DNA has mutated,it still has its own bound
	for i,dna in enumerate(child):
		if np.random.rand() < MUTATION_RATE:
			#so each one mutate on its own way
			if i == 0:
				dna = np.random.rand()/10
			elif i == 2:
				dna = np.random.randint(3,100,1)
			elif i == 5:
				dna = np.random.rand()
			else:
				dna = np.random.randint(0,100,1)
	return child

def create():
	#create a module for DNA
	dna_mod = np.empty((POP_SIZE,DNA_SIZE))
	#each DNA has its own bound
	for i in range(POP_SIZE):
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
	
	return dna_mod

if __name__ == '__main__':
	starttime = datetime.datetime.now()
	np.seterr(divide='ignore',invalid='ignore')
	pop = create()
	max_found = -9999
	for _ in range(N_GENERATION+1):
		
		fitness = get_fitness(pop)
		if _ % 10 == 0:
			length = fitness.shape[0]
			#print(length)
			x = range(length)
			plt.plot(x,fitness)
			plt.show()
			if _ == N_GENERATION:
				break
		
		#find the maxMatch params
		idx = fitness.argmax()
		if(fitness[idx] > max_found):
			max_found = fitness[idx]
			max_param = pop[idx][:]

		print("max match:",max_found)
		print("params:",max_param)

		for i,item in enumerate(fitness):
			if item < (-5000):
				fitness[i] = 0

		pop = select(pop,fitness)

		pop_copy = pop.copy()

		for parent in pop:
			child = crossover(parent,pop_copy)
			child = mutate(child)
			parent = child

	endtime = datetime.datetime.now()
	print('runtime:',(endtime - starttime).seconds)
