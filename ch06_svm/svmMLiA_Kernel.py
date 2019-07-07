from numpy import *
# SMO算法中的辅助函数
def loadDataSet(filename):
	dataMat=[]
	labelMat=[]
	fr = open(filename)
	for line in fr.readlines():
		lineArr = line.strip().split('\t')
		dataMat.append([float(lineArr[0]), float(lineArr[1])]) #二维
		labelMat.append(float(lineArr[2]))
	return dataMat, labelMat

def selectJrand(i, m):
	"""产生不等于i的随机数"""
	j = i
	while(j == i):
		j = int(random.uniform(0,m))
	return j

def clipAlpha(aj, H, L):
	"""用于调整大于H或小于L的alpha值"""
	if aj > H:
		aj = H
	if(aj < L):
		aj = L
	return aj

# 完整版Platt SMO
# 支持函数: 1个用于清理代码的数据结构和3个对E缓存的辅助函数
def kernelTrans(X,A,kTup):
	"""核转换函数"""
	m, n = shape(X)
	K = mat(zeros((m,1)))
	if kTup[0] == 'lin':
		K = X * A.T
	elif kTup[0] == 'rbf':
		for j in range(m):
			deltaRow = X[j,:] - A
			K[j] = deltaRow * deltaRow.T
		# K = exp(K/(-2*kTup[1]**2))
		K = exp(K / (-1 * kTup[1] ** 2))
	else:
		raise NameError("Kernel is not recognized")
	return K

class optStruct:
	""""清理数据结构"""
	def __init__(self, dataMatIn, classLabels, C, toler,kTup):
		self.X = dataMatIn
		self.labelMat = classLabels
		self.C = C
		self.tol = toler
		self.m = shape(dataMatIn)[0]
		self.alphas = mat(zeros((self.m, 1)))
		self.b = 0
		self.eCache = mat(zeros((self.m,2))) # 第一列 是否有效的标志位，第二列 实际E值
		self.K = mat(zeros((self.m, self.m)))
		for i in range(self.m):
			self.K[:,i] = kernelTrans(self.X, self.X[i,:], kTup)#第i列是X与第i个向量的内积
																#第i行是第i个向量与X的内积

def clacEk(oS, k):
	# fx_k = float(multiply(oS.alphas, oS.labelMat).T *(oS.X * oS.X[k,:].T)) + oS.b
	fx_k = float(multiply(oS.alphas, oS.labelMat).T * oS.K[:,k] + oS.b)
	E_k = float(fx_k - oS.labelMat[k])
	return E_k

def selectJ(i, oS, Ei):
	"""内循环中的启发式方法：选择改变最大的那个值"""
	maxK = -1; maxDeltaE = 0; Ej = 0
	oS.eCache[i] = [1,Ei]
	validEcacheList = nonzero(oS.eCache[:,0].A)[0] #有效即计算好的
	if (len(validEcacheList)) > 1:
		for k in validEcacheList:
			if k == i:
				continue
			Ek = clacEk(oS, k)
			deltaE = abs(Ei-Ek)
			if(deltaE > maxDeltaE):
				maxK, maxDeltaE, Ej = k, deltaE, Ek
		return maxK, Ej
	else:
		# 首次循环，随机选择一个alpha
		j = selectJrand(i, oS.m)
		Ej = clacEk(oS, j)
	return j, Ej

def updateEk(oS, k):
	Ek = clacEk(oS, k)
	oS.eCache[k] = [1,Ek]

# 优化过程
def innerL(i, oS):
	"""使用了自己的数据结构的smo函数"""
	E_i = clacEk(oS, i)
	if ((oS.labelMat[i] * E_i < -oS.tol) and (oS.alphas[i] < oS.C)) or \
			(oS.alphas[i] > 0 and oS.labelMat[i] * E_i > oS.tol):
		j, E_j = selectJ(i, oS, E_i)
		alpha_iold = oS.alphas[i].copy()
		alpha_jold = oS.alphas[j].copy()
		# 保证alpha_j在0 C之间
		if (oS.labelMat[i] != oS.labelMat[j]):
			L = max(0, oS.alphas[j] - oS.alphas[i])
			H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
		else:
			L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
			H = min(oS.C, oS.alphas[i] + oS.alphas[j])
		# if l==h 不做任何改变，直接返回0
		if L == H:
			print("L == H")
			return 0
		eta = 2.0 * oS.K[i,j] - oS.K[i,i] - oS.K[j,j]
		# 简化 if 最优修改量eta==0 直接返回0
		if eta >= 0:
			print("eta>=0")
			return 0
		# 递推公式
		oS.alphas[j] -= oS.labelMat[j] * (E_i - E_j) / eta
		oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)

		if abs(oS.alphas[j] - alpha_jold) < 1e-5:
			print("j not moving enough")
			return 0
		# 对i进行修改，修改量与j相同，但方向相反
		# delta_i * yi = - delta_j*yj
		oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alpha_jold - oS.alphas[j])
		updateEk(oS, i)
		# 给alphai alphaj设置常数项
		b1 = oS.b - E_i - \
		     oS.labelMat[i] * (oS.alphas[i] - alpha_iold) * oS.K[i,i] - \
		     oS.labelMat[j] * (oS.alphas[j] - alpha_jold) * oS.K[i,j]
		b2 = oS.b - E_j - \
		     oS.labelMat[i] * (oS.alphas[i] - alpha_iold) * oS.K[i,j]- \
		     oS.labelMat[j] * (oS.alphas[j] - alpha_jold) * oS.K[j,j]
		if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
			oS.b = b1
		elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
			oS.b = b2
		else:
			oS.b = (b1 + b2) / 2.0
		return 1
	else:
		return 0

# 外循环
def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):
	oS = optStruct(mat(dataMatIn), mat(classLabels).transpose(),C,toler,kTup)
	iter = 0
	entireSet = True; alphaPairsChanged = 0
	# 迭代次数超过指定最大值或者遍历整个集合都未对任意alpha进行修改
	while(iter <maxIter) and ((alphaPairsChanged > 0) or(entireSet)):
		alphaPairsChanged = 0
		# 在完整遍历和非边界循环之间进行切换
		if entireSet:
			for i in range(oS.m):
				alphaPairsChanged += innerL(i, oS)
				print("fullSet, iter: %d i: %d, pairs changed：%d" %(iter, i, alphaPairsChanged))
			iter += 1
		else:
			nonBoundIs = nonzero((oS.alphas.A > 0)*(oS.alphas.A < C))[0] #筛选非边界alpha
			for i in nonBoundIs:
				alphaPairsChanged += innerL(i, oS)
				print("non-bound, iter: %d i: %d, pairs changed：%d" %(iter, i, alphaPairsChanged))
			iter += 1
		if entireSet:
			entireSet = False
		elif alphaPairsChanged == 0:
			entireSet = True
		print("iteration number: %d" %iter)
	return oS.b, oS.alphas

def testRbf(k1 = 1.3):
	"""利用核函数进行分类的径向基测试函数"""
	dataArr,labelArr = loadDataSet('testSetRBF.txt')
	b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1))
	print(b)
	print(alphas[alphas > 0])
	dataMat = mat(dataArr); labelMat = mat(labelArr).transpose()
	# 构建支持向量矩阵
	svInd = nonzero(alphas.A > 0)[0] # 支持向量的索引
	sVs = dataMat[svInd]
	labelSV = labelMat[svInd]
	print("there are %d Support Vector" % shape(sVs)[0])
	m, n = shape(dataMat)
	errorCount = 0
	for i in range(m):
		# 仅利用支持向量数据进行分类
		kernalEval = kernelTrans(sVs, dataMat[i,:], ('rbf', k1))
		predict = kernalEval.T * multiply(labelSV, alphas[svInd]) + b
		if sign(predict)!=sign(labelArr[i]):
			errorCount += 1
	print("the training error rate is: %f" %(float(errorCount)/m) )

if __name__ == "__main__":
	testRbf()