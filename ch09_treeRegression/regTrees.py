from numpy import *


class TreeNode():
	# 树节点
	def __init__(self,feat,val,right,left):
		featureToSplitOn = feat
		valueOfSplit = val
		rightBranch = right
		leftBranch = left

def loadDataSet(filename):
	dataMat = []
	fr = open(filename)
	for line in fr.readlines():
		curLine = line.strip().split('\t')
		fltLine = list(map(float,curLine))
		dataMat.append(fltLine)
	return dataMat

def regLeaf(dataSet):#returns the value used for each leaf
	leaf =  mean(dataSet[:,-1])
	print(leaf)
	return leaf

def regErr(dataSet):
	"""平方误差"""
	return var(dataSet[:,-1]) * shape(dataSet)[0]

def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
	"""
	用最佳方式二元切分数据集合生成相应的叶节点
	:param dataSet:
	:param leafType:创建叶节点的函数
	:param errType:数据一致性（混乱度）：默认为总方差
	:param ops:tolS是容许的误差下降值， tolN是切分的最少样本数
	:return:特征编号和切分特征值，如果无则产生叶节点返回none
	"""
	tolS = ops[0] # 容许的误差下降值
	tolN = ops[1] # 切分的最少样本数
	if len(set(dataSet[:,-1].T.tolist()[0])) == 1: #如果所有值相等则退出
		return None, leafType(dataSet)
	m,n = shape(dataSet)
	S = errType(dataSet)
	bestS = inf; bestIndex = 0; bestValue = 0
	# 对每个特征，对每个特征值
	for featIndex in range(n-1): #最后一个是类别
		for splitVal in set(dataSet[:,featIndex].T.tolist()[0]):
			# 将数据集切分成两份
			mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
			if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
				continue # 小于切分最少样本数
			# 计算切分的误差
			newS = errType(mat0) + errType(mat1)
			# 更新最佳切分点
			if newS < bestS:
				bestIndex = featIndex
				bestValue = splitVal
				bestS = newS
	if (S - bestS) < tolS:
		return None, leafType(dataSet) #如果误差减小不大（小于容许值）则退出
	mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
	if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
		return None, leafType(dataSet) #如果切分出的数据集很小则退出
	return bestIndex, bestValue

def binSplitDataSet(dataSet,feature,value):
	"""
	通过数组过滤方式将数据集切分得到两个子集返回
	:param dataSet: 数据集合
	:param feature: 待切分特征
	:param value: 该特征的某个值
	:return: 切分后的两个子集
	"""
	mat0 = dataSet[nonzero(dataSet[:,feature] > value)[0],:]
	mat1 = dataSet[nonzero(dataSet[:,feature] <= value)[0],:]
	return mat0, mat1

def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
	"""
	递归地创建树
	:param dataSet:数据集
	:param leafType:建立叶节点的函数（默认平均法
	:param errType: 误差计算函数（默认方差
	:param ops:树构建所需其他参数的元组
	:return:一棵树
	"""
	feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
	if feat == None: return val #返回叶节点
	retTree = {} #树的数据结构是字典
	retTree['spInd'] = feat
	retTree['spVal'] = val
	lSet, rSet = binSplitDataSet(dataSet, feat, val)
	retTree['left'] = createTree(lSet, leafType, errType, ops)
	retTree['right'] = createTree(rSet, leafType, errType, ops)
	return retTree

# 回归树后剪枝
def isTree(obj):
	return type(obj).__name__ == 'dict'

def getMean(tree):
	"""对树进行塌陷处理，返回树平均值"""
	if isTree(tree['right']):
		tree['right'] = getMean(tree['right'])
	if isTree(tree['left']):
		tree['left'] = getMean(tree['left'])
	return (tree['left'] + tree['right'])/2.0


### 以下未测试和使用
def prune(tree,testData):
	"""后剪枝函数"""
	# 测试集是否为空, 为空则对树进行塌陷处理
	if shape(testData)[0] == 0:
		return getMean(tree)
	# 基于已有的树切分测试数据
	if (isTree(tree['right']) or isTree(tree['left'])):
		lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
		# 对是树的子集进行递归剪枝
		if isTree(tree['left']):
			tree['left'] = prune(tree['left'], lSet)
		if isTree(tree['right']):
			tree['right'] = prune(tree['right'], rSet)
	# 左右子树剪枝后仍然检查是否是子树，是叶节点则可以进行合并操作
	if not isTree(tree['left']) and not isTree(tree['right']):
		# 找到叶节点，用测试集判断将叶节点合并是否能降低误差
		lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
		# 计算当前两个叶节点合并后的误差
		errorNoMerge = sum(power(lSet[:,-1]-tree['left'] ,2)) + \
						sum(power(rSet[:,-1]-tree['right'],2))
		treeMean = (tree['left'] + tree['right'])/2.0
		# 计算不合并的误差
		errorMerge = sum(power(testData[:,-1]-treeMean,2))
		if errorMerge < errorNoMerge:
			print("merging")
			return treeMean
		else:
			return tree
	else:
		return tree

# 模型树
def linearSolve(dataSet):
	"""格式化数据集，计算回归系数ws"""
	m, n = shape(dataSet)
	X = mat(ones((m,n))); Y = mat(ones((m,1)))
	X[:,1:n] = dataSet[:, 0:n-1] # x0 = 1, x(1:n)=dat(0:n-1)
	Y = dataSet[:,-1]
	xTx = X.T*X
	if linalg.det(xTx) == 0.0:
		raise NameError("this matrix is singular, cannot do inverse,\n try increasing the second value of the ops")
	ws = xTx*(X.T*Y)
	return ws, X, Y

def modelLeaf(dataSet):
	"""模型树的叶节点创建:返回回归系数"""
	ws, X, Y = linearSolve(dataSet)
	return ws

def modelErr(dataSet):
	"""模型树计算数据集上的误差：计算yHat和Y之间的平方误差"""
	ws, X, Y = linearSolve(dataSet)
	yHat = X * ws
	return sum(power(Y-yHat),2)

# 模型树：createTree(dataSet, modelLeaf, modelErr, (1,10))

def main():
	myDat = loadDataSet('ex00.txt')
	myDat = mat(myDat)
	print(createTree(myDat))


if __name__ == "__main__":
	main()