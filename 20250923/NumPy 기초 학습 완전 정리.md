# NumPy 기초 학습 완전 정리

## 목차
1. [NumPy 소개 및 기본 설정](#1-numpy-소개-및-기본-설정)
2. [배열 생성 및 기본 속성](#2-배열-생성-및-기본-속성)
3. [특수 배열 생성 함수](#3-특수-배열-생성-함수)
4. [배열 인덱싱 및 슬라이싱](#4-배열-인덱싱-및-슬라이싱)
5. [배열 연산](#5-배열-연산)
6. [통계 함수 및 집계 연산](#6-통계-함수-및-집계-연산)
7. [axis 개념 심화 이해](#7-axis-개념-심화-이해)
8. [실습 예제 및 주요 포인트](#8-실습-예제-및-주요-포인트)

---

## 1. NumPy 소개 및 기본 설정

### NumPy란?
- **N**umerical **Py**thon의 줄임말
- 파이썬에서 수치 계산을 위한 핵심 라이브러리
- 고성능 다차원 배열 객체와 배열 처리 도구 제공

### 기본 import
```python
import numpy as np  # 관례적으로 np로 alias 사용
```

---

## 2. 배열 생성 및 기본 속성

### 2.1 리스트에서 배열 생성

```python
# 1차원 배열
a_1d = [1, 2, 3, 4, 5]
arr_1d = np.array(a_1d)
print(arr_1d)  # [1 2 3 4 5]

# 2차원 배열 (행렬)
a_2d = [[1, 2, 3], [4, 5, 6]]
arr_2d = np.array(a_2d)
print(arr_2d)
# 출력:
# [[1 2 3]
#  [4 5 6]]
```

### 2.2 배열의 기본 속성

```python
a = [[1, 2, 3], [4, 5, 6]]
b = np.array(a)

# 데이터 타입 확인
print(type(a))        # <class 'list'>
print(type(b))        # <class 'numpy.ndarray'>

# 차원(dimension) 확인
print(b.ndim)         # 2 (2차원 배열)

# 배열의 형태(shape) 확인
print(b.shape)        # (2, 3) → 2행 3열

# 전체 요소 개수
print(b.size)         # 6

# 데이터 타입
print(b.dtype)        # int64 (시스템에 따라 다를 수 있음)
```

### 2.3 배열 요소 접근 (인덱싱)

```python
b = np.array([[1, 2, 3], [4, 5, 6]])

# 단일 요소 접근
print(b[0, 0])        # 1 (첫 번째 행, 첫 번째 열)
print(b[1, 2])        # 6 (두 번째 행, 세 번째 열)
print(b[0][1])        # 2 (리스트 방식으로도 접근 가능, 비추천)

# 음수 인덱스 사용
print(b[-1, -1])      # 6 (마지막 행, 마지막 열)
```

---

## 3. 특수 배열 생성 함수

### 3.1 영행렬 생성 (zeros)

```python
# 2차원 영행렬
c = np.zeros((2, 2))
print(c)
# 출력:
# [[0. 0.]
#  [0. 0.]]

# 1차원 영배열
d = np.zeros(5)
print(d)              # [0. 0. 0. 0. 0.]

# 3차원 영배열
e = np.zeros((2, 3, 4))  # 2×3×4 배열
print(e.shape)        # (2, 3, 4)
```

### 3.2 일행렬 생성 (ones)

```python
# 3×2 일행렬
f = np.ones((3, 2))
print(f)
# 출력:
# [[1. 1.]
#  [1. 1.]
#  [1. 1.]]

# 데이터 타입 지정
g = np.ones((2, 3), dtype=int)
print(g)
# 출력:
# [[1 1 1]
#  [1 1 1]]
```

### 3.3 특정 값으로 채운 배열 (full)

```python
# 2×3 배열을 5로 채우기
h = np.full((2, 3), 5)
print(h)
# 출력:
# [[5 5 5]
#  [5 5 5]]

# 다른 값으로 채우기
i = np.full((3, 2), 3.14)
print(i)
# 출력:
# [[3.14 3.14]
#  [3.14 3.14]
#  [3.14 3.14]]
```

### 3.4 기타 유용한 생성 함수

```python
# 단위행렬 (항등행렬)
identity = np.eye(3)
print(identity)
# 출력:
# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]]

# 연속된 수 배열
seq = np.arange(10)      # 0부터 9까지
print(seq)               # [0 1 2 3 4 5 6 7 8 9]

seq2 = np.arange(2, 10, 2)  # 2부터 10미만까지 2씩 증가
print(seq2)              # [2 4 6 8]

# 균등 분할
linear = np.linspace(0, 1, 5)  # 0부터 1까지 5개로 균등 분할
print(linear)            # [0.   0.25 0.5  0.75 1.  ]
```

---

## 4. 배열 인덱싱 및 슬라이싱

### 4.1 기본 슬라이싱

```python
lst = [[1, 2, 3],
       [4, 5, 6], 
       [7, 8, 9]]
arr = np.array(lst)
print("원본 배열:")
print(arr)
# 출력:
# [[1 2 3]
#  [4 5 6]
#  [7 8 9]]

# 부분 배열 추출
a = arr[0:2, 0:2]    # 첫 2행, 첫 2열
print("arr[0:2, 0:2]:")
print(a)
# 출력:
# [[1 2]
#  [4 5]]

b = arr[1:, 1:]      # 2행부터 끝까지, 2열부터 끝까지
print("arr[1:, 1:]:")
print(b)
# 출력:
# [[5 6]
#  [8 9]]
```

### 4.2 고급 슬라이싱

```python
arr = np.array([[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]])

# 전체 행, 특정 열들
print(arr[:, [0, 2]])    # 모든 행의 1열과 3열
# 출력:
# [[ 1  3]
#  [ 5  7]
#  [ 9 11]]

# 특정 행들, 전체 열
print(arr[[0, 2], :])    # 1행과 3행의 모든 열
# 출력:
# [[ 1  2  3  4]
#  [ 9 10 11 12]]

# 특정 행과 특정 열
print(arr[[0, 2], [1, 3]])  # (0,1)과 (2,3) 위치의 요소들
# 출력: [ 2 12]
```

---

## 5. 배열 연산

### 5.1 요소별 연산 (Element-wise Operations)

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# 덧셈
c_add = a + b                    # 또는 np.add(a, b)
print("덧셈:", c_add)            # [5 7 9]

# 뺄셈  
c_sub = a - b                    # 또는 np.subtract(a, b)
print("뺄셈:", c_sub)            # [-3 -3 -3]

# 곱셈 (요소별)
c_mul = a * b                    # 또는 np.multiply(a, b)
print("요소별 곱셈:", c_mul)      # [4 10 18]

# 나눗셈
c_div = a / b                    # 또는 np.divide(a, b)
print("나눗셈:", c_div)          # [0.25 0.4  0.5 ]

# 거듭제곱
c_pow = a ** 2                   # 또는 np.power(a, 2)
print("제곱:", c_pow)            # [1 4 9]
```

### 5.2 브로드캐스팅 (Broadcasting)

```python
# 배열과 스칼라 연산
arr = np.array([[1, 2, 3],
                [4, 5, 6]])
result = arr + 10
print("배열 + 스칼라:")
print(result)
# 출력:
# [[11 12 13]
#  [14 15 16]]

# 서로 다른 크기 배열 연산
a = np.array([[1, 2, 3],
              [4, 5, 6]])     # 2×3
b = np.array([10, 20, 30])    # 1×3

result = a + b
print("브로드캐스팅 연산:")
print(result)
# 출력:
# [[11 22 33]
#  [14 25 36]]
```

### 5.3 행렬 곱셈

```python
# 2×2 행렬들의 곱셈
arr1 = [[1, 2], 
        [3, 4]]
arr2 = [[5, 6], 
        [7, 8]]

a = np.array(arr1)
b = np.array(arr2)

# 행렬 곱셈 (dot product)
c = np.dot(a, b)              # 또는 a @ b
print("행렬 곱셈 결과:")
print(c)
# 출력:
# [[19 22]    # [1*5+2*7, 1*6+2*8]
#  [43 50]]   # [3*5+4*7, 3*6+4*8]

# 요소별 곱셈과 비교
elementwise = a * b
print("요소별 곱셈 결과:")
print(elementwise)
# 출력:
# [[ 5 12]
#  [21 32]]
```

---

## 6. 통계 함수 및 집계 연산

### 6.1 기본 통계 함수

```python
a = np.array([[-1, 2, 3], 
              [3, 4, 8]])

print("원본 배열:")
print(a)
# 출력:
# [[-1  2  3]
#  [ 3  4  8]]

# 전체 합계
print("전체 합계:", a.sum())           # 19

# 평균
print("평균:", a.mean())               # 3.1666...

# 표준편차
print("표준편차:", a.std())            # 2.943...

# 분산
print("분산:", a.var())                # 8.666...

# 최댓값과 최솟값
print("최댓값:", a.max())              # 8
print("최솟값:", a.min())              # -1

# 모든 요소의 곱
print("전체 곱:", a.prod())            # -576
```

### 6.2 축별 연산 (axis 매개변수)

```python
a = np.array([[-1, 2, 3], 
              [3, 4, 8]])

# axis=0: 세로 방향 (행끼리 연산, 열별 결과)
print("열별 합계 (axis=0):", a.sum(axis=0))     # [2 6 11]
print("열별 평균 (axis=0):", a.mean(axis=0))    # [1. 3. 5.5]

# axis=1: 가로 방향 (열끼리 연산, 행별 결과)  
print("행별 합계 (axis=1):", a.sum(axis=1))     # [4 15]
print("행별 평균 (axis=1):", a.mean(axis=1))    # [1.333... 5.]
```

### 6.3 최댓값/최솟값 위치 찾기

```python
a = np.array([[-1, 2, 3], 
              [3, 4, 8]])

# 전체에서 최댓값/최솟값 위치
print("최댓값 위치:", a.argmax())       # 5 (flatten된 인덱스)
print("최솟값 위치:", a.argmin())       # 0 (flatten된 인덱스)

# 축별 최댓값/최솟값 위치
print("열별 최댓값 위치:", a.argmax(axis=0))  # [1 1 1]
print("행별 최댓값 위치:", a.argmax(axis=1))  # [2 2]
```

---

## 7. axis 개념 심화 이해

### 7.1 axis 개념 시각화

```
2차원 배열에서:
axis=0 ↓ (세로 방향, 행들을 따라 이동)
axis=1 → (가로 방향, 열들을 따라 이동)

예시 배열:
    열0 열1 열2
행0 [ 1,  2,  3]  ← axis=1 방향
행1 [ 4,  5,  6]  ← axis=1 방향
    ↑   ↑   ↑
  axis=0 방향
```

### 7.2 3차원 배열에서의 axis

```python
# 3차원 배열 생성 (2×3×4)
arr_3d = np.arange(24).reshape(2, 3, 4)
print("3차원 배열 형태:", arr_3d.shape)  # (2, 3, 4)

print("axis=0으로 합계:", arr_3d.sum(axis=0).shape)  # (3, 4)
print("axis=1으로 합계:", arr_3d.sum(axis=1).shape)  # (2, 4)  
print("axis=2으로 합계:", arr_3d.sum(axis=2).shape)  # (2, 3)
```

---

## 8. 실습 예제 및 주요 포인트

### 8.1 종합 실습 예제

```python
import numpy as np

# 1. 5×4 랜덤 배열 생성
data = np.random.rand(5, 4)
print("랜덤 데이터:")
print(data)

# 2. 각 행의 평균 계산
row_means = data.mean(axis=1)
print("각 행의 평균:", row_means)

# 3. 각 열의 최댓값 계산
col_maxs = data.max(axis=0)
print("각 열의 최댓값:", col_maxs)

# 4. 전체 데이터의 표준편차
overall_std = data.std()
print("전체 표준편차:", overall_std)

# 5. 조건부 선택 (0.5보다 큰 값들만)
large_values = data[data > 0.5]
print("0.5보다 큰 값들:", large_values)
```

### 8.2 성능 비교 예제

```python
import time

# 파이썬 리스트 vs NumPy 배열 성능 비교
size = 1000000

# 파이썬 리스트
python_list = list(range(size))
start_time = time.time()
python_sum = sum([x**2 for x in python_list])
python_time = time.time() - start_time

# NumPy 배열
numpy_array = np.arange(size)
start_time = time.time()
numpy_sum = np.sum(numpy_array**2)
numpy_time = time.time() - start_time

print(f"Python 리스트 소요 시간: {python_time:.4f}초")
print(f"NumPy 배열 소요 시간: {numpy_time:.4f}초")
print(f"NumPy가 {python_time/numpy_time:.1f}배 더 빠름")
```

---

## 📚 핵심 포인트 정리

### 🔹 기억해야 할 핵심 개념

1. **배열 vs 리스트**
   - NumPy 배열: 같은 데이터 타입, 고성능, 벡터화 연산
   - Python 리스트: 다양한 데이터 타입, 유연성, 상대적으로 느림

2. **axis 이해**
   ```
   axis=0: 첫 번째 차원 (행 방향)
   axis=1: 두 번째 차원 (열 방향)
   axis=2: 세 번째 차원 (깊이 방향)
   ```

3. **연산의 종류**
   - **요소별 연산**: `*`, `+`, `-`, `/` → 같은 위치 요소끼리 연산
   - **행렬 곱셈**: `np.dot()`, `@` → 수학적 행렬 곱셈 규칙

4. **브로드캐스팅**
   - 서로 다른 크기의 배열 간 연산 가능
   - 작은 배열이 큰 배열의 크기에 맞춰 자동 확장

### 🔹 자주 실수하는 부분

1. **인덱싱 방식**
   ```python
   # 좋은 방식
   arr[i, j]
   
   # 피해야 할 방식
   arr[i][j]  # 성능상 비효율적
   ```

2. **복사 vs 뷰**
   ```python
   # 뷰 생성 (원본과 메모리 공유)
   view = arr[1:3, 1:3]
   
   # 복사본 생성
   copy = arr[1:3, 1:3].copy()
   ```

3. **데이터 타입 주의**
   ```python
   # 정수 나눗셈 주의
   a = np.array([1, 2, 3], dtype=int)
   result = a / 2  # float64로 자동 변환
   ```

### 🔹 실무에서 유용한 패턴

1. **배열 초기화 패턴**
   ```python
   # 크기가 정해진 배열 미리 생성 후 값 채우기
   result = np.zeros((n, m))
   for i in range(n):
       result[i] = some_computation()
   ```

2. **조건부 연산**
   ```python
   # where 함수 활용
   result = np.where(condition, true_value, false_value)
   
   # 마스킹 활용
   mask = arr > threshold
   arr[mask] = new_value
   ```

3. **배열 변형**
   ```python
   # reshape: 형태 변경
   arr.reshape(new_shape)
   
   # flatten: 1차원으로 평탄화
   arr.flatten()
   
   # transpose: 전치
   arr.T
   ```

---

## 🚀 다음 학습 추천 주제

1. **고급 인덱싱** - 불리언 인덱싱, 팬시 인덱싱
2. **배열 결합 및 분할** - concatenate, split, stack
3. **선형대수** - 고유값, 고유벡터, 역행렬
4. **파일 I/O** - 배열 저장/로드 (save, load, savetxt)
5. **무작위 수 생성** - random 모듈 활용

---

*이 문서는 NumPy 기초 학습을 위한 완전 가이드입니다. 각 예제를 직접 실행해보며 학습하시기 바랍니다.*