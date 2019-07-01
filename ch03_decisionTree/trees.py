from numpy import *
from math import log
import collections

def calcShannonEnt(dataSet):
	"""计算给定数据集的信息熵"""
	numEntries = len(dataSet)
	labelCounts = {}
	for featVec in dataSet:
		currentLabel = featVec[-1]
		if currentLabel not in labelCounts.keys():
			labelCounts[currentLabel] = 0
		labelCounts[currentLabel] += 1
	shannonEnt = 0.0
	for key in labelCounts:
		prob = float(labelCounts[key])/numEntries
		shannonEnt -= prob*log(prob, 2)
	return shannonEnt

def splitDataSet(dataSet, axis, value):
	"""
	按照给定特征划分数据集
	:param dataSet: 待划分的数据集
	:param axis: 划分数据集的特征
	:param value: 特征的返回值
	:return:
	"""
	retDataSet = []  # 创建新对象，不修改原数据集
	for featVec in dataSet:
		if(featVec[axis] == value):
			reducedFeatVec = featVec[:axis]
			reducedFeatVec.extend(featVec[axis+1:]) #去除此属性
			retDataSet.append(reducedFeatVec)
	return retDataSet

def chooseBestFeatureToSplit(dataSet):
	"""
	遍历整个数据集，循环计算熵和splitDataSet函数，找到最好的数据集划分方式
	:param dataSet:
	:return: bestFeature
	"""
	numFeatures = len(dataSet[0])-1 #总特征数，减去类别的1
	baseEntropy = calcShannonEnt(dataSet)
	bestInfoGain = 00.0
	bestFeature = -1
	for i in range(numFeatures):
		# 创建唯一的分类标签列表
		featList = [example[i] for example in dataSet] #所有样本的第i个特征组成的列表
		uniqueVals = set(featList) # 不同取值的集合
		# 计算每种划分方式的信息熵
		newEntropy = 0.0
		for value in uniqueVals:
			subDataSet = splitDataSet(dataSet, i, value)
			prob = len(subDataSet)/float(len(dataSet))
			newEntropy += prob * calcShannonEnt(subDataSet)
		infoGain = baseEntropy - newEntropy
		# 计算最好的信息增益
		if (infoGain > bestInfoGain):
			bestInfoGain = infoGain
			bestFeature = i
	return bestFeature

def majorityCnt(classList):
	"""
	类似于投票表决代码
	:param classList:分类名称的列表
	:return: 返回出现次数最多的分类名称
	"""
	return collections.Counter(classList).most_common(1)[0][0]

def createTree(dataSet,labels):
	"""递归的创建树"""
	classList = [example[-1] for example in dataSet] #数据集中所有的类标签
	#
	if classList.count(classList[0]) == len(classList):
		return classList[0]
	#
	if len(dataSet[0]) == 1:
		return majorityCnt(classList)
	#
	bestFeat = chooseBestFeatureToSplit(dataSet)
	bestFeatLabel = labels[bestFeat]
	myTree = {bestFeatLabel:{}}
	del(labels[bestFeat])
	featValues = [example[bestFeat] for example in dataSet]
	uniqueFeatValues = set(featValues)
	for value in uniqueFeatValues:
		myTree[bestFeatLabel][value] = createTree(
			splitDataSet(dataSet,bestFeat,value), labels[:])
	return myTree
def createDataSet():
	dataSet = [[1, 1, 'yes'],
	           [1, 0, 'no'],
	           [0, 1, 'no'],
	           [0, 1, 'no']]
	labels = ['no surfacing', 'flippers']
	return dataSet, labels

def classify(inputTree, featLabels, testVec):
	"""
	使用决策树的分类函数：比较 测试数据 与 决策树上节点的数值，直至到达叶节点
	:param inputTree:
	:param featLabels:特征标签列表，借助index比较属性值
	:param testVec:属性顺序与特征标签列表一致
	:return:分类标签
	"""
	firstStr = list(inputTree.keys())[0]
	secondDict = inputTree[firstStr]
	featIndex = featLabels.index(firstStr) # 将标签字符串转换为索引
	for key in secondDict.keys():
		if testVec[featIndex] == key:
			if type(secondDict[key]).__name__ == 'dict':
				classLabel = classify(secondDict[key], featLabels,testVec)
			else:
				classLabel = secondDict[key]
	return classLabel

# 使用pickle模块存储决策树
def storeTree(inputTree, filename):
	import pickle
	# 序列化对象
	fw = open(filename, 'w')
	pickle.dump(inputTree, fw)
	fw.close()


def grabTree(filename):
	import pickle
	fr = open(filename)
	return pickle.load(fr)

def main():
	myDat, labels = createDataSet()
	print(chooseBestFeatureToSplit(myDat))
	print(majorityCnt([1,2,3,4,1,4,1]))
	print(createTree(myDat,labels))

if __name__ == "__main__":
	main()