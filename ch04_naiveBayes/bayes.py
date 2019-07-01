
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



def main():
	listOPosts, listClasses = loadDataSet()
	myVocabList = createVocabList(listOPosts)
	print(myVocabList)
	print(setOfWords2Vec(myVocabList,listOPosts[0]))


if __name__ == "__main__":
	main()