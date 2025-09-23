# -*- coding: utf-8 -*-
import numpy as np # numpy 패키지 로드하여 np로 사용
a = [[1,2,3], [4,5,6]] # 리스트에서 행렬생성 , 2차 행렬
b = np.array(a) #[[1 2 3] 
                # [4 5 6]]
print(b)
print(type(b))
print(type(a))
print(b.ndim) # 차원
print(b.shape) # 행렬의 크기

print(b[0,0]) # 1 배열의 원소접근
print(b[1,2]) # 6

# 배열의 생성
c = np.zeros((2,2)) # 0으로 채워진 2행2열 배열생성
print(c) # [[0. 0.]
         #  [0. 0.]]
d = np.zeros(5) # 0으로 채워진 1차원 배열생성
print(d) # [0. 0. 0. 0. 0.]
e = np.ones((3,2)) # 1로 채워진 3행2열 배열생성
print(e) # [[1. 1.]
         #  [1. 1.]
         #  [1. 1.]]
f = np.full((2,3),5) # 5로 채워진
print(f) # [[5 5 5]
         #  [5 5 5]]
