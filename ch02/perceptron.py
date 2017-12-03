import numpy as np

def AND(x1, x2) :
	x = np.array([x1, x2]) # 입력 값
	w = np.array([0.5, 0.5]) # 가중치
	b = -0.7 # 편향 
	tmp = np.sum(w*x) + b
	if tmp <= 0 :
		return 0
	else :
		return 1

#NAND는 AND의 반대
def NAND(x1, x2) :
	x = np.array([x1, x2])
	w = np.array([-0.5, -0.5])
	b = 0.7
	tmp = np.sum(w*x) + b
	if tmp <= 0 :
		return 0
	else :
		return 1 

def OR(x1 ,x2) :
	x = np.array([x1, x2])
	w = np.array([0.5, 0.5])
	b = -0.2
	tmp = np.sum(w*x) + b
	if tmp <= 0 :
		return 0
	else :
		return 1 

# AND, NAND, OR은 모두 같은 구조의 퍼셉트론이고 서로 가중치의 매개변수의 값만 다르다

# XOR - NAND와 OR, AND의 조합으로 만들 수 있다.

def XOR(x1, x2) :
	s1 = NAND(x1, x2)
	s2 = OR(x1, x2)
	y = AND(s1, s2)
	return y

print(XOR(0,0))
print(XOR(0,1))
print(XOR(1,0))
print(XOR(1,1))
