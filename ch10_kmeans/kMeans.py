from numpy import *

# 支持函数
def loadDataSet(filename):
	dataMat = []
	fr = open(filename)
	for line in fr.readlines():
		currLine = line.strip().split('\t')
		fltLine = list(map(float, currLine))
		dataMat.append(fltLine)
	return dataMat

def distEclud(vecA, vecB):
	return sqrt(sum(power(vecA-vecB, 2)))

def randCent(dataSet,k):
	"""随机构建簇质心"""
	n = shape(dataSet)[1]
	centroids = mat(zeros((k,n)))
	for j in range(n):
		minJ = min(array(dataSet[:,j]))
		maxJ = max(array(dataSet[:,j]))
		centroids[:,j] = minJ + (maxJ-minJ)*random.rand(k,1) #[0,1)
	return centroids

# k均值聚类算法
def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
	"""
	k均值聚类
	:param dataSet:
	:param k:
	:param distMeas: 距离计算方法，默认欧拉
	:param createCent: 创建k个初始质心的方法
	:return: k个质心， 簇分配结果（簇索引值和误差
	"""
	m = shape(dataSet)[0] # 样本数
	clusterAssment = mat(zeros((m,2)))
	centroids = createCent(dataSet, k)
	clusterChanged = True
	while clusterChanged:
		clusterChanged = False
		for i in range(m):
			# 寻找最近的质心
			minDist = inf
			minIndex = -1
			for j in range(k):
				distji = distEclud(centroids[j,:], dataSet[i,:])
				if(distji < minDist):
					minDist = distji
					minIndex = j
			if clusterAssment[i,0] != minIndex:
				clusterChanged = True
			clusterAssment[i,:] = minIndex, minDist**2 #簇标记，平方误差
		print(centroids)
		# 更新k个质心的位置
		for cent in range(k):
			# mat->array, int->bool->index_array
			ptsInClust = dataSet[nonzero(clusterAssment[:,0].A == cent)[0]]
			# 计算新的均值向量
			centroids[cent,:] = mean(ptsInClust,axis=0)
	return centroids, clusterAssment

def biKmeans(dataSet, k, distMeas=distEclud):
	m = shape(dataSet)[0]
	clusterAssment = mat(zeros((m,2)))
	# 创建一个包含全部样本的初始簇,计算簇中心
	centroid0 = mean(dataSet, axis=0).tolist()[0]
	centList =[centroid0] #保存质心的列表
	for j in range(m):
		clusterAssment[j,1] = distMeas(mat(centroid0), dataSet[j,:])**2
	while(len(centList) < k):
		lowestSSE = inf
		# 尝试划分每一簇，基于SSE选择
		for i in range(len(centList)):
			ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A == i)[0],:]
			centroidMat, splitClustAss = kMeans(ptsInCurrCluster,2,distMeas)
			sseSplit = sum(splitClustAss[:,1])
			sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A != i)[0],1])
			print("sseSplit, sseNotSplit is :", sseSplit, sseNotSplit)
			if (sseNotSplit + sseSplit) < lowestSSE:
				bestCentToSplit = i
				bestNewCents = centroidMat
				bestClustAss = splitClustAss.copy()
				lowestSSE = sseSplit + sseNotSplit
		# 更新簇的分配结果
		bestClustAss[nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList) # 新增簇
		bestClustAss[nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit # 分割簇
		print("the bestCentToSplit is: ",bestCentToSplit)
		print("the len of bestClustAss is: ",len(bestClustAss))
		centList[bestCentToSplit] = bestNewCents[0,:] #更新质心
		centList.append(bestNewCents[1,:]) #更新质心
		# 更新所分割的簇的样本的簇标记和平方误差
		clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:]=bestClustAss
	return centList, clusterAssment

def main():
	dataMat = mat(loadDataSet('testSet.txt'))
	myCentroids, clustAssing = kMeans(dataMat,4)
	dataMat2 = mat(loadDataSet('testSet2.txt'))
	centList, myNewAssments = biKmeans(dataMat2,3)
	print(centList)

# 解析地图数据及分类

def distSLC(vecA, vecB):
	"""球面距离计算"""
	a = sin(vecA[0,1]*pi/180) * sin(vecB[0,1]*pi/180) #sin(a)sin(b)
	b = cos(vecA[0,1]*pi/180) * cos(vecB[0,1]*pi/180) #cos(a)cos(b)
	return arccos(a+b) * 6371.0

import matplotlib
import matplotlib.pyplot as plt
def clusterClubs(numClust=5):
	"""对地理坐标进行聚类和进行簇绘图"""
	pass


if __name__ == "__main__":
	main()