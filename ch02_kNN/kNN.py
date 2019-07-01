from numpy import *
import operator

def createDataSet():
	group = array([[1.0, 1.1], [ 1.0, 1.0], [0.0, 0.0], [0, 0.1]])
	labels = ['A', 'A', 'B', 'B']
	return group, labels

def classify0(inX, dataSet, labels, k):
	"""
	inX:用于分类的向量
	dataSet:训练样本集
	labels:标签向量
	k:最近邻居的数目
	:return:
	"""
	dataSetSize = dataSet.shape[0]
	# 已知类别点与当前点的距离，采用欧氏距离
	diffMat = tile(inX, (dataSetSize, 1)) - dataSet
	sqDiffMat = diffMat ** 2
	sqDistances = sqDiffMat.sum(axis=1)
	distances = sqDistances ** 0.5
	# 按距离排序,返回索引
	sortedDistIndicies = distances.argsort()
	# 确定前k元素所在的主要分类
	classCount = {}
	for i in range(k):
		voteIlabel = labels[sortedDistIndicies[i]]
		classCount[voteIlabel]= classCount.get(voteIlabel,0) + 1
	# 运算符模块的itemgetter方法,按照频率排序
	sortedClassCount = sorted(classCount.items(),
	                          key = operator.itemgetter(1), reverse=True)
	return sortedClassCount[0][0]

def main():
	group, labels = createDataSet()
	print(classify0([0,0], group,labels,3))


if __name__ == "__main__":
	main()