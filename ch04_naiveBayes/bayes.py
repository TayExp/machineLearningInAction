from numpy import *
def loadDataSet():
	"""

	:return:进行词条切分后的文档集合， 类别标签的集合
	"""
	postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
	               ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
	               ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
	               ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
	               ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
	               ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
	classVec = [0,1,0,1,0,1] #1 代表侮辱性文字， 0 代表正常言论
	return postingList, classVec

def createVocabList(dataSet):
	vocabSet = set([])
	for document in dataSet:
		vocabSet |= set(document) # 集合求并
	return list(vocabSet) #返回不重复词表

def setOfWords2Vec(vocabList, inputSet):
	"""
	词表到向量的转换函数：遍历文档中所有单词，看文档中包含哪些词汇表单词
	:param vocabList:词汇表
	:param inputSet:某个文档
	:return:文档向量(每个元素为0或1，表示词汇表中的单词是否出现在输入文档中)
	"""
	returnVec = [0] * len(vocabList)
	for word in inputSet:
		if word in vocabList:
			returnVec[vocabList.index(word)] = 1
		else:
			print( "the word %s is not in my Vocabulary" % word)
	return returnVec

def trainNB0(trainMatrix, trainCategory):
	"""
	朴素贝叶斯分类器训练函数
	:param trainMatrix: 文档矩阵
	:param trainCategory: 每篇文档类别标签组成的向量
	:return:
	"""
	numTrainDocs = len(trainMatrix)
	numWords = len(trainMatrix[0])
	pAbusive = sum(trainCategory)/numTrainDocs
	p0Num = ones(numWords)
	p1Num = ones(numWords)
	p0Denom = 2.0 #总数
	p1Denom = 2.0
	for i in range(numTrainDocs):
		if trainCategory[i] == 1:
			p1Num += trainMatrix[i]
			p1Denom += sum(trainMatrix[i])
		else:
			p0Num += trainMatrix[i]
			p0Denom += sum(trainMatrix[i])
	p1Vect = log(p1Num/p1Denom)
	p0Vect = log(p0Num/p0Denom)
	return p0Vect, p1Vect, pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
	p1 = sum(vec2Classify*p1Vec) + log(pClass1) # 元素相乘，log相加
	p0 = sum(vec2Classify*p0Vec) + log(1-pClass1)
	if(p1 > p0):
		return 1
	else:
		return 0


def main():
	listOPosts, listClasses = loadDataSet()
	myVocabList = createVocabList(listOPosts)
	print(myVocabList)
	print(setOfWords2Vec(myVocabList,listOPosts[0]))
	#
	trainMat=[]
	for postingDoc in listOPosts:
		trainMat.append(setOfWords2Vec(myVocabList,postingDoc))
	p0V, p1V, pAb = trainNB0(trainMat, listClasses)
	print(p0V)
	print(p1V)
	print(pAb)
	# testingNB
	testingEntry = ['love', 'my', 'dalmation']
	thisDoc = array(setOfWords2Vec(myVocabList, testingEntry))
	print(testingEntry, 'classified as: ', classifyNB(thisDoc,p0V,p1V,pAb))
	testingEntry = ['stupid','garbage']
	thisDoc = array(setOfWords2Vec(myVocabList, testingEntry))
	print(testingEntry, 'classified as: ', classifyNB(thisDoc,p0V,p1V,pAb))

def bagOfWords2VecMN(vocabList, inputSet):
	"""朴素贝叶斯磁带模型"""
	returnVec = [0] * len(vocabList)
	for word in inputSet:
		if word in vocabList:
			returnVec[vocabList.index(word)] += 1
	return returnVec

def textParse(bigString):
	"""文件解析"""
	import re
	listTokens = re.split(r'\w*', bigString)
	return [tok.lower for tok in listTokens if len(tok) > 2]

def spamTest():
	"""垃圾邮件测试函数"""
	docList = []
	classList = []
	fullText = []
	for i in range(1, 26):
		wordList = textParse(open('email/spam/%d.txt' % i, encoding='ISO-8859-1').read())
		docList.append(wordList)
		fullText.extend(wordList)
		classList.append(1)
		wordList = textParse(open('email/ham/%d.txt' % i, encoding='ISO-8859-1').read())
		docList.append(wordList)
		fullText.extend(wordList)
		classList.append(0)
	vocabList = createVocabList(docList)
	trainingSet = list(range(50))
	testSet=[]
	for i in range(10):
		randIndex = int(random.uniform(0,len(trainingSet)))#不包含hi？
		testSet.append(trainingSet[randIndex])
		del(trainingSet[randIndex]) #原因：python3.x range返回的是range对象，不返回数组对象
	trainMat = []
	trainClass = []
	for docIndex in trainingSet:
		trainMat.append(setOfWords2Vec(vocabList,docList[docIndex]))
		trainClass.append(classList[docIndex])
	p0V,p1V,pSpam = trainNB0(array(trainMat), array(trainClass))
	errorCount = 0
	for docIndex in testSet:
		wordVector = setOfWords2Vec(vocabList, docList[docIndex])
		if classifyNB(array(wordVector), p0V,p1V,pSpam) != classList[docIndex]:
			errorCount += 1
	print('the error rate is: ', float(errorCount)/len(testSet))

if __name__ == "__main__":
	#main()
	spamTest()