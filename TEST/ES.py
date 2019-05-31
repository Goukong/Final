#To complete with c++
import numpy as np 
import cv2
import targetTacing
import datetime
import matplotlib.pyplot as plt

DNA_SIZE =  6
DNA_BOUND1 = [0.01,0.1] # for qualityLevel
DNA_BOUND2 = [0,100] # for minDistance,winSize,maxlevel,COUNT
DNA_BOUND3 = [0,1] #for EPS
N_GENERATIONS = 20 #this time the param will be used in c++
POP_SIZE = 100
N_KID = 50   
MAX_CORNERS = 1000

def get_fitness(x):
	result = np.empty(x.shape[0])
	for _ in range(x.shape[0]):
		feature_params = dict(maxCorners = MAX_CORNERS,
			qualityLevel = x[_][0],
			minDistance = x[_][1],
			blockSize = 3
			)
		lk_params = dict(winSize = (int(x[_][2]),int(x[_][2])),
			maxLevel = int(x[_][3]),
			criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,int(x[_][4]),x[_][5])
			)
		mse = targetTacing.targetTrace(feature_params,lk_params)
		result[_] = mse
	return result

def make_kid(pop,n_kid):
	kids = {'DNA':np.empty((n_kid,DNA_SIZE))}
	kids['mut_strength'] = np.empty_like(kids['DNA'])
	for kv,ks in zip(kids['DNA'],kids['mut_strength']):
		p1,p2 = np.random.choice(np.arange(POP_SIZE),size=2,replace=False)
		cp = np.random.randint(0,2,size=DNA_SIZE,dtype=np.bool)
		kv[cp] = pop['DNA'][p1,cp]
		kv[~cp] = pop['DNA'][p2,~cp]
		ks[cp] = pop['mut_strength'][p1,cp]
		ks[~cp] = pop['mut_strength'][p2,~cp]

		#mutate
		ks[:] = np.maximum(ks+(np.random.randn(*ks.shape)),0.0) 
		kv += ks * np.random.randn(*kv.shape)
		kv[0] = np.clip(kv[0],*DNA_BOUND1)
		kv[1] = np.clip(kv[1],*DNA_BOUND2)
		kv[2] = np.clip(kv[2],3,100)
		kv[3:5] = np.clip(kv[3:5],*DNA_BOUND2)
		kv[5] = np.clip(kv[5],*DNA_BOUND3)
		
	return kids 

def kill_bad(pop,kids):
	#把原一代和新一代连接起来
	for key in ['DNA','mut_strength']:
		pop[key] = np.vstack((pop[key],kids[key]))
	fitness = get_fitness(pop['DNA'])
	idx = np.arange(pop['DNA'].shape[0])#POP_SIZE + n_kids
	good_idx = idx[fitness.argsort()][-POP_SIZE:]#要最大的POP_SIZE个,多余的为空，在赋值的时候直接舍去
	
	for key in ['DNA','mut_strength']:
		pop[key] = pop[key][good_idx]#已经kill掉n_kids的数量
	return pop

np.seterr(divide='ignore',invalid='ignore')
starttime = datetime.datetime.now()

dna_mod = np.empty((POP_SIZE,DNA_SIZE))
for i in range(POP_SIZE):
	dna_mod[i][0] = np.random.rand()/10 #qualityLevel
	dna_mod[i][1] = np.random.randint(0,100,1)
	dna_mod[i][2] = np.random.randint(3,100,1)
	dna_mod[i][3] = np.random.randint(0,100,1)
	dna_mod[i][4] = np.random.randint(0,100,1)
	dna_mod[i][5] = np.random.rand()

pop = dict(DNA=dna_mod,
	mut_strength=np.random.rand(POP_SIZE,DNA_SIZE)*3)
 

for i in range(N_GENERATIONS+1):
	kids = make_kid(pop,N_KID)
	pop = kill_bad(pop,kids)
	result = get_fitness(pop['DNA'])
	if i%10 == 0:
		length = result.shape[0]
		x = range(length)
		plt.plot(x,result)
		plt.show()
		if i == N_GENERATIONS:
			break
	print(max(result))
	idx = np.argmax(result)
	best_choice = pop['DNA'][idx]
	print(best_choice)

endtime = datetime.datetime.now()
print('runtime:',(endtime - starttime).seconds)
