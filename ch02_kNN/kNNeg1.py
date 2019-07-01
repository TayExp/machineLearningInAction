from numpy import *
import matplotlib
import matplotlib.pyplot as plt
from ch02_kNN.kNN import *
def file2matrix(filename):
	"""
	将文本记录转换为numpy矩阵
	:param filename:
	:return:
	"""
	fr = open(filename)
	arrayOLines = fr.readlines()
	numberOfLines = len(arrayOLines)
	returnMat = zeros((numberOfLines,3))
	# 解析文件数据到列表
	classLabelVector = []
	index = 0
	for line in arrayOLines:
		line = line.strip()
		listFromLine = line.split('\t')
		returnMat[index,:] = listFromLine[0:3]
		classLabelVector.append(int(listFromLine[-1]))
		index += 1
	return returnMat, classLabelVector

def autoNorm(dataSet):
	"""
	归一化数值
	:param dataSet: original data set
	:return: norm data set, Norm parameters
	"""
	minVals = dataSet.min(0) # 向下 按列
	maxVals = dataSet.max(0)
	ranges = maxVals - minVals
	normDataSet = zeros(shape(dataSet))
	m = dataSet.shape[0] #数据个数
	normDataSet = dataSet - tile(minVals, (m,1))
	normDataSet = normDataSet/tile(ranges, (m,1))
	return normDataSet, ranges, minVals

def datingClassTest():
	"""分类器针对约会网站的测试代码"""
	hoRatio = 0.10 #测试集比例
	datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
	normMat, ranges, minVals = autoNorm(datingDataMat)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2],
	           15.0 * array(datingLabels), 15.0 * array(datingLabels))
	plt.show()
	m = normMat.shape[0]
	numTestVecs = int(m*hoRatio)
	errorCount = 0.0
	for i in range(numTestVecs):
		classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m,:], datingLabels[numTestVecs:m],3)
		print("the classifier came back with: %d, the real answer is : %d" %(classifierResult,datingLabels[i]))
		if(classifierResult != datingLabels[i]):
			errorCount += 1.0
	print("the total error rate is : %f" %(errorCount/float(numTestVecs)))

def classifyPerson():
	"""约会网站预测函数：加入允许用户输入文本行命令的功能"""
	pass

def main():
	datingClassTest()

if __name__ == "__main__":
	main()