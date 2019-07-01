import matplotlib.pyplot as plt
import numpy
# 定义文本框和箭头格式
decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
	createPlot.ax1.annotate(nodeTxt, xy=parentPt,xycoords='axes fraction',
	                        xytext=centerPt, textcoords='axes fraction',
	                        arrowprops=arrow_args,bbox=nodeType,
	                        va="center",ha="center")

# 获取叶节点的数目
def getNumLeaf(myTree):
	numLeafs = 0
	firstStr = list(myTree.keys())[0] # 标签
	secondDict = myTree[firstStr]
	for key in  secondDict.keys():
		if type(secondDict[key]).__name__ == 'dict':
			numLeafs += getNumLeaf(secondDict[key])
		else:
			numLeafs += 1
	return numLeafs

# 获取树的层数
def getTreeDepth(myTree):
	maxDepth = 0
	firstStr = list(myTree.keys())[0]  # 标签
	secondDict = myTree[firstStr]
	for key in secondDict.keys():
		if type(secondDict[key]).__name__ == 'dict':
			thisDepth = 1 + getTreeDepth(secondDict[key])
		else:
			thisDepth = 1
		maxDepth = max(maxDepth, thisDepth)
	return maxDepth

def retrieveTree(i):
	"""输出预先存储的树的信息，避免每次都用创建树"""
	listOfTrees =[{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                  {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}]
	return listOfTrees[i]

# plotTree函数
def plotMidText(centerPt, parentPt, txtString):
	xMid = (parentPt[0]-centerPt[0])/2.0 + centerPt[0]
	yMid = (parentPt[1]-centerPt[1])/2.0 + centerPt[1]
	createPlot.ax1.text(xMid, yMid, txtString)


def plotTree(myTree, parentPt, nodeTxt):
	numLeafs = getNumLeaf(myTree)
	depth = getTreeDepth(myTree)
	firstStr = list(myTree.keys())[0]
	# x坐标偏移 （本身1 + 其下叶节点）
	centerPt = (plotTree.xOff +
	            (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)
	plotMidText(centerPt, parentPt, nodeTxt)
	plotNode(firstStr, centerPt, parentPt, decisionNode)
	secondDict = myTree[firstStr]
	# y坐标下移，绘子节点
	plotTree.yOff -= 1.0/plotTree.totalD
	for key in secondDict.keys():
		if type(secondDict[key]).__name__ == 'dict':
			plotTree(secondDict[key], centerPt, str(key))
		else:
			plotTree.xOff += 1.0/plotTree.totalW #兄弟叶节点，右移一小格
			plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), centerPt, leafNode)
			plotMidText((plotTree.xOff, plotTree.yOff), centerPt, str(key))
	# 绘完子节点，y坐标返回
	plotTree.yOff += 1.0 / plotTree.totalD


def createPlot(intree):
	fig = plt.figure(1, facecolor='white')
	fig.clf()
	axprops = dict(xticks=[], yticks=[])
	createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
	plotTree.totalW = float(getNumLeaf(intree))
	plotTree.totalD = float(getTreeDepth(intree))
	print(plotTree.totalW,plotTree.totalD)

	plotTree.xOff = -0.5/plotTree.totalW
	plotTree.yOff = 1.0
	print(plotTree.xOff, plotTree.yOff)
	plotTree(intree, (0.5,1.0), '')
	plt.show()


def main():
	print(retrieveTree(1))
	myTree = retrieveTree(0)
	print(getNumLeaf(myTree))
	print(getTreeDepth(myTree))
	createPlot(retrieveTree(0))

if __name__ == "__main__":
	main()