# -*- coding: utf-8 -*-
import numpy as np
a = np.array([[-1,2,3],[3,4,8]])
s = np.sum(a)
print('sum=',a.sum()) 

# 행별/열별 연산 (axis=0/1)

print('sum by row=',a.sum(axis=0)) # 위 아래 방향 계산

print('sum by col=',a.sum(axis=1)) # 좌 우 방향 계산

print('mean=',a.mean()) # 평균

print('sd=',a.std()) # 표준편차

print('product=',a.prod()) # 모든 요소 곱