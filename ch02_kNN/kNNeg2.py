from numpy import *
import os
from ch02_kNN.kNN import classify0
def img2vector(filename):
	returnVect = zeros((1,1024))
	fr = open(filename)
	for i in range(32):
		lineStr = fr.readline()
		for j in range(32):
			# 00000000000000011000000000000000
			returnVect[0,32*i+j] = int(lineStr[j])
	return returnVect

def handwritingClassTest():
	"""手写数字识别系统"""
	trainingFileList = os.listdir('trainingDigits') # 获取目录内容
	m = len(trainingFileList)
	hwLabels = []
	trainingMat = zeros((m, 1024))
	for i in range(m):
		fileNameStr = trainingFileList[i]
		fileStr = fileNameStr.split('.')[0] #获取文件名
		classNumStr = int(fileStr.split('_')[0]) # 从文件名中获取label
		hwLabels.append(classNumStr)
		trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
	# 测试数据
	testFileList = os.listdir('testDigits')
	errorCount = 0.0
	mTest = len(testFileList)
	for i in range(mTest):
		fileNameStr = testFileList[i]
		fileStr = fileNameStr.split('.')[0]
		classNumStr = int(fileStr.split('_')[0])
		vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
		classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
		print("the classifier came back with: %d, the real answer is : %d" % (classifierResult, classNumStr))
		if (classifierResult != classNumStr):
			errorCount += 1.0
	print("the total error rate is : %f" % (errorCount / float(mTest)))

def main():
	handwritingClassTest()

if __name__ == "__main__":
	main()