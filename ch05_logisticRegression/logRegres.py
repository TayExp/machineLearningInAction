from numpy import *

def loadDataSet():
	dataMat = []
	labelMat = []
	fr = open('testSet.txt')
	for line in fr.readlines():
		lineArr = line.strip().split()
		dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
		labelMat.append(int(lineArr[2]))
	return dataMat, labelMat

def sigmoid(inX):
	return 1.0/(1+exp(-inX))

def gradAscent(dataMatIn, classLabels):
	"""
	梯度上升算法
	:param dataMatIn:样本数据集
	:param classLabels: 类别标签向量
	:alpha 步长 maxCycles 迭代次数
	:return: 回归系数
	"""
	dataMatrix = mat(dataMatIn) #每行是一个样本
	labelMat = mat(classLabels).transpose()#转换为列向量
	m, n = shape(dataMatrix) #样本数， 特征数
	alpha = 0.001
	maxCycles = 500
	weights = ones((n,1))# 每个特征的回归系数，初始均设为1
	for k in range(maxCycles):
		h = sigmoid(dataMatrix*weights)
		error = (labelMat - h) #列向量
		weights += alpha * dataMatrix.transpose()*error #梯度上升 n*m  *  m*1
	return weights

def plotBestFit(weights):
	"""画出数据集合logistic回归最佳拟合直线的函数"""
	import matplotlib.pyplot as plt
	# weights = wei.getA() # 变成ndarray， 相当于np.asarray(self)
	dataMat, labelMat = loadDataSet()
	dataArr = array(dataMat)
	n = shape(dataArr)[0]
	xcord1 = []
	ycord1 = []
	xcord2 = []
	ycord2 = []
	for i in range(n):
		if int(labelMat[i]) == 1:
			xcord1.append(dataArr[i,1])
			ycord1.append(dataArr[i,2])
		else:
			xcord2.append(dataArr[i,1])
			ycord2.append(dataArr[i,2])
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
	ax.scatter(xcord2, ycord2,s=30,c='green')
	x = arange(-3,3,0.1)
	y=(-weights[0]-weights[1]*x) / weights[2] #最佳拟合曲线 w0*x0+w1*x1+w2*y = 0
	ax.plot(x,y)
	plt.xlabel('X1')
	plt.ylabel('X2')
	plt.show()

def stocGradAscent0(dataMatrix, classLabels):
	"""随机梯度上升算法"""
	m, n = shape(dataMatrix)
	alpha = 0.01
	weights = ones(n) # 不需要转换为mat，仍为数组
	for i in range(m):
		h = sigmoid(sum(dataMatrix[i] * weights))
		error = classLabels[i]-h  # 标量
		weights += alpha * error * dataMatrix[i]
	return weights

def stocGradAscent1(dataMatrix, classLabels, numIter = 150):
	"""改进的随机梯度上升算法"""
	m,n = shape(dataMatrix)
	weights = ones(n)
	for j in range(numIter):
		dataIndex = list(range(m))
		for i in range(m):
			# 改进1：调整alpha，缓解数据波动或高频波动；且alpha不会减小到0,且不严格下降
			alpha = 4/(1.0+j+i) + 0.01
			#改进2: 随机选取样本更新回归系数，减少周期性波动
			randIndex = int(random.uniform(0,len(dataIndex)))
			h = sigmoid(sum(dataMatrix[randIndex]*weights))
			error = classLabels[randIndex]-h
			weights += alpha*error * dataMatrix[randIndex]
			del(dataIndex[randIndex])
	return weights

def main():
	dataArr, labelMat = loadDataSet()
	wei = gradAscent(dataArr, labelMat)
	plotBestFit(wei)

# 从疝气病症预测病马的死亡率
def classifyVector(inX, weights):
	"""

	:param inX: 特征向量
	:param weights: 回归系数
	:return: 预测类别
	"""
	prop = sigmoid(sum(inX * weights))
	return 1.0 if prop > 0.5 else 0.0

def colicTest():
	# 对于此问题：缺失值用0替换，缺失标签则丢弃
	frTrain = open('horseColicTraining.txt')
	frTest = open('horseColicTest.txt')
	trainingSet = []
	trainingLabels = []
	for line in frTrain.readlines():
		currLine = line.strip().split('\t')
		lineArr = []
		for i in range(21): #21个特征
			lineArr.append(float(currLine[i]))
		trainingSet.append(lineArr)
		trainingLabels.append(float(currLine[21]))
	trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 500)
	errorCount = 0
	numTestVec = 0.0
	for line in frTest.readlines():
		numTestVec += 1
		currLine = line.strip().split('\t')
		lineArr = []
		for i in range(21):
			lineArr.append(float(currLine[i]))
		if int(classifyVector(array(lineArr), trainWeights) != int(currLine[21])):
			errorCount += 1
	errorRate = (float(errorCount)/numTestVec)
	return errorRate

def multiTest():
	numTests = 10
	errorSum = 0.0
	for k in range(numTests):
		errorSum += colicTest()
	print("after %d iterations the average error rate is: %f" %(numTests, errorSum/float(numTests)))


if __name__ == "__main__":
	main()
	multiTest()