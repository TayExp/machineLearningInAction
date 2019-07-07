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

def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
	"""
	简化版SMO算法
	:param dataMatIn:数据集
	:param classLabels: 类别标签
	:param C: 常数
	:param toler: 容错率
	:param maxIter: 取消前的最大循环次数
	:return: b, alphas
	"""
	dataMatrix = mat(dataMatIn)
	labelMat = mat(classLabels).transpose()
	b = 0
	m, n = shape(dataMatrix)
	alphas = mat(zeros((m,1)))
	iter = 0
	while(iter < maxIter):
		alphaPairsChanged = 0 # 记录alpha是否已经进行优化
		# 对整个集合进行遍历
		for i in range(m):
			# fxi= sum(ai*yi*xiT) * x + b
			fx_i = float(multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[i,:].T)) + b
			E_i = fx_i - float(labelMat[i])
			# 如果误差较大 且 a不等于0和C 进行优化
			if ((labelMat[i]*E_i < -toler) and (alphas[i] < C) ) or \
				(alphas[i] > 0 and labelMat[i]*E_i > toler):
			# if abs(labelMat[i]*E_i) > toler and alphas[i] != 0 and alphas[i] != C:
				j = selectJrand(i, m)
				fx_j = float(multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[j,:].T)) + b
				E_j = fx_j - float(labelMat[j])
				alpha_iold = alphas[i].copy()
				alpha_jold = alphas[j].copy()
				# 保证alpha_j在0 C之间
				if(labelMat[i] != labelMat[j]):
					L = max(0, alphas[j]-alphas[i])
					H = min(C, C+alphas[j]-alphas[i])
				else:
					L = max(0, alphas[j]+alphas[i]-C)
					H = min(C,alphas[i]+alphas[j])
				# if l==h 不做任何改变，直接执行continue语句
				if L == H:
					print("L == H")
					continue
				eta = 2.0 * dataMatrix[i,:] * dataMatrix[j,:].T - \
							dataMatrix[i,:] * dataMatrix[i,:].T - \
							dataMatrix[j,:] * dataMatrix[j,:].T
				# 简化 if 最优修改量eta==0 continue
				if eta >= 0:
					print("eta>=0")
					continue
				# 递推公式
				alphas[j] -= labelMat[j] * (E_i- E_j)/eta
				alphas[j] = clipAlpha(alphas[j],H,L)
				if abs(alphas[j]-alpha_jold) < 1e-5:
					print("j not moving enough")
					continue
				#对i进行修改，修改量与j相同，但方向相反
				# delta_i * yi = - delta_j*yj
				alphas[i] += labelMat[j] * labelMat[i] *(alpha_jold-alphas[j])
				#给alphai alphaj设置常数项
				b1 = b - E_i - \
				     labelMat[i] * (alphas[i]-alpha_iold)*dataMatrix[i,:]*dataMatrix[i,:].T - \
					 labelMat[j] * (alphas[j]-alpha_jold)*dataMatrix[i,:]*dataMatrix[j,:].T
				b2 = b - E_j - \
					 labelMat[i] * (alphas[i]-alpha_iold)*dataMatrix[i,:]*dataMatrix[j,:].T - \
					 labelMat[j] * (alphas[j]-alpha_jold)*dataMatrix[j,:]*dataMatrix[j,:].T
				if (0 < alphas[i]) and (C > alphas[i]): b = b1
				elif (0 < alphas[j]) and (C > alphas[j]): b = b2
				else: b = (b1+b2)/2.0
				alphaPairsChanged += 1
				print("iter: %d i: %d,pairs changed %d"%(iter, i ,alphaPairsChanged))
		if (alphaPairsChanged == 0):
			iter += 1
		else:
			iter = 0
		print("iteration number: %d" % iter)
	return b, alphas


# 完整版Platt SMO
# 支持函数: 1个用于清理代码的数据结构和3个对E缓存的辅助函数
class optStruct:
	""""清理数据结构"""
	def __init__(self, dataMatIn, classLabels, C, toler):
		self.X = dataMatIn
		self.labelMat = classLabels
		self.C = C
		self.tol = toler
		self.m = shape(dataMatIn)[0]
		self.alphas = mat(zeros((self.m, 1)))
		self.b = 0
		self.eCache = mat(zeros((self.m,2))) # 第一列 是否有效的标志位，第二列 实际E值

def clacEk(oS, k):
	fx_k = float(multiply(oS.alphas, oS.labelMat).T *(oS.X * oS.X[k,:].T)) + oS.b
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
		eta = 2.0 * oS.X[i, :] * oS.X[j, :].T - \
		      oS.X[i, :] * oS.X[i, :].T - oS.X[j, :] * oS.X[j, :].T
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
		     oS.labelMat[i] * (oS.alphas[i] - alpha_iold) * oS.X[i, :] * oS.X[i, :].T - \
		     oS.labelMat[j] * (oS.alphas[j] - alpha_jold) * oS.X[i, :] * oS.X[j, :].T
		b2 = oS.b - E_j - \
		     oS.labelMat[i] * (oS.alphas[i] - alpha_iold) * oS.X[i, :] * oS.X[j, :].T - \
		     oS.labelMat[j] * (oS.alphas[j] - alpha_jold) * oS.X[j, :] * oS.X[j, :].T
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
	oS = optStruct(mat(dataMatIn), mat(classLabels).transpose(),C,toler)
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

def main():
	dataArr,labelArr = loadDataSet('testSet.txt')
	b, alphas = smoSimple(dataArr, labelArr, 0.6, 0.001, 40)
	print(b)
	print(alphas[alphas>0])
	b1, alphas1 = smoP(dataArr, labelArr, 0.6, 0.001, 40)
	print(b1)
	print(alphas1[alphas1 > 0])

def calcWs(alphas, dataArr, classLabels):
	"""基于alpha值得到超平面，w的计算"""

	X = mat(dataArr)
	labelMat = mat(classLabels).transpose()
	m,n = shape(X)
	w = zeros((n,1))
	for i in range(m):
		w += multiply(alphas[i]*labelMat[i], X[i,:].T)
	return w
# x * mat(w) + b


if __name__ == "__main__":
	main()