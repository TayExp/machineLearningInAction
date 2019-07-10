from numpy import *
def loadSimpleData():
	datMat = matrix([[1., 2.1],
					 [2., 1.1],
					 [1.3, 1.],
					 [1., 1.],
					 [2., 1.]])
	classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
	return datMat,classLabels

def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
	"""
	通过阈值比较对数据进行分类。是否有某个值大于 或者 小于等于该阈值
	:param dataMatrix:
	:param dimen: 第几个特征
	:param threshVal: 阈值
	:param threshIneq: 'gt', lt' = less than or equal
	:return: 符合threshIneq为-1默认为1的数组
	"""
	retArray = ones((shape(dataMatrix)[0],1))
	if threshIneq == 'lt':
		retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
	else:
		retArray[dataMatrix[:,dimen] > threshVal] = -1.0
	return retArray

def buildStump(dataArr, classLabels, D):
	"""
	遍历所有可能值，找到数据集上的最佳单层决策树:每个特征，每个值，gt和lt
	:param dataArr:
	:param classLabels:
	:param D:
	:return:
	"""
	dataMatrix = mat(dataArr)
	labelMat = mat(classLabels).T
	m,n = shape(dataMatrix)
	numSteps = 10.0 #每个特征遍历10个值
	bestStump = {}
	bestClasEst = mat(zeros((m,1)))
	minError = inf
	# 三层循环
	for i in range(n):
		rangeMin = dataMatrix[:,i].min()# 第i特征的最小值
		rangeMax = dataMatrix[:,i].max()
		stepSize = (rangeMax - rangeMin) /numSteps
		for j in range(-1,int(numSteps)+1):
			for inequal in['lt', 'gt']:
				threshVal = (rangeMin + float(j)*stepSize)
				# 计算错误率
				predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)
				errArr = mat(ones((m,1)))
				errArr[predictedVals == labelMat] = 0
				weightedError = D.T*errArr
				if weightedError < minError:
					minError = weightedError
					bestClasEst = predictedVals.copy()
					bestStump['dim'] = i
					bestStump['thresh'] = threshVal
					bestStump['ineq'] = inequal
	return bestStump, minError, bestClasEst


def adaBoostTrainDS(dataArr, classLabels, numIt=40):
	"""基于单层决策树(decision stump)的AdaBoost训练过程"""
	# 利用buildStump找到最佳单层决策树
	weakClassArr = []
	m = shape(dataArr)[0]
	D = mat(ones((m,1))/1.0/m)
	aggClassEst = mat(zeros((m,1)))
	for i in range(numIt):
		bestStump, error, classEst = buildStump(dataArr, classLabels, D)
		print("D:",D.T)
		# 计算alpha
		alpha = float(0.5 * log((1.0 - error) / max(error, 1e-16)))
		bestStump['alpha'] = alpha
		weakClassArr.append(bestStump)
		print("classEst:", classEst.T)
		# 计算新的权重向量D
		expon = multiply(-1*alpha*mat(classLabels).T,classEst)
		D = multiply(D, exp(expon))
		D = D/D.sum()
		# 更新累计类别估计值
		aggClassEst += alpha*classEst
		print("aggClassEst:",aggClassEst.T)
		aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m,1)))
		errorRate = aggErrors.sum()/m
		print("total error: ",errorRate)
		# 如果错误率等于0，则退出循环
		if errorRate == 0.0: break
	return weakClassArr

def main():
	D = mat(ones((5,1))/5.0)
	dataMat, classLabels = loadSimpleData()
	print(buildStump(dataMat,classLabels,D))
	classifierArr = adaBoostTrainDS(dataMat,classLabels,9)
	adaClassify([0, 0], classifierArr )

# 测试算法：基于AdaBoost的分类
def adaClassify(datToClass, classifierArr):
	"""利用训练出的多个弱分类器进行分类"""
	dataMatrix = mat(datToClass)
	m = shape(dataMatrix)[0]
	aggClassEst = mat(zeros((m,1)))
	for i in range(len(classifierArr)):
		classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'],
								 classifierArr[i]['thresh'],classifierArr[i]['ineq'])
		aggClassEst += classifierArr[i]['alpha']*classEst
		print(aggClassEst)
	return sign(aggClassEst)

if __name__ == "__main__":
	main()
