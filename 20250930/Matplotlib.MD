# Matplotlib 기초 학습 정리

## 📚 목차
1. [Matplotlib 소개](#matplotlib-소개)
2. [기본 플롯 생성](#기본-플롯-생성)
3. [선 그래프 (Line Plot)](#선-그래프-line-plot)
4. [막대 그래프 (Bar Chart)](#막대-그래프-bar-chart)
5. [실전 예제](#실전-예제)

---

## Matplotlib 소개

Matplotlib은 Python에서 가장 많이 사용되는 데이터 시각화 라이브러리입니다.

```python
import matplotlib.pyplot as plt
import numpy as np
```

---

## 기본 플롯 생성

### 가장 간단한 플롯
```python
plt.plot([2, 3, 4, 5], linewidth=1, c='b')
plt.show()
```

- `plt.plot()`: 그래프를 그리는 기본 함수
- `plt.show()`: 그래프를 화면에 표시

---

## 선 그래프 (Line Plot)

### 1. 제목 설정 (Title)

```python
plt.title("Graph Title", 
          fontsize=20,           # 글자 크기
          color='green',         # 색상
          fontweight='bold',     # 굵기
          loc='left',            # 위치 (left, center, right)
          pad=10)                # 여백
```

### 2. 선 스타일 (Line Styles)

```python
# 실선
plt.plot([1,2,3], [4,5,6], linestyle='-')

# 점선
plt.plot([1,2,3], [4,5,6], linestyle='--')

# 대시-점 선
plt.plot([1,2,3], [4,5,6], linestyle='-.')

# 점으로 이루어진 선
plt.plot([1,2,3], [4,5,6], linestyle=':')
```

**주요 선 스타일:**
- `-` : 실선 (solid)
- `--` : 긴 대시선 (dashed)
- `-.` : 대시-점선 (dash-dot)
- `:` : 점선 (dotted)

### 3. 선 색상과 두께

```python
plt.plot([1,2,3], [4,5,6], 
         c='green',        # 색상
         linewidth=2)      # 선 두께
```

**색상 지정 방법:**
- 이름: `'red'`, `'blue'`, `'green'`
- 약자: `'r'`, `'b'`, `'g'`
- RGB: `'#FF0000'`

### 4. 격자선 (Grid)

```python
plt.grid(linestyle='--')  # 점선 스타일의 격자선
```

### 5. 축 레이블 (Axis Labels)

```python
plt.xlabel('Sequence', 
           fontsize=10, 
           color='green', 
           fontweight='bold')

plt.ylabel('Time(secs)', 
           fontsize=10, 
           color='green', 
           fontweight='bold')
```

### 6. 범례 (Legend)

```python
plt.legend(['Mouse', 'Cat'],    # 레이블 리스트
           loc='upper left',     # 위치
           fontsize=10)
```

**범례 위치 옵션:**
- `'upper left'`, `'upper right'`
- `'lower left'`, `'lower right'`
- `'center'`, `'best'` (자동으로 최적 위치)

### 7. 축 범위 설정 (Axis Limits)

```python
plt.xlim([0, 4])   # X축 범위: 0 ~ 4
plt.ylim([0, 10])  # Y축 범위: 0 ~ 10
```

---

## 막대 그래프 (Bar Chart)

### 기본 막대 그래프

```python
y = [5, 3, 7, 10, 9, 5, 3.5, 8]
x = range(len(y))  # 0, 1, 2, 3, 4, 5, 6, 7

plt.bar(x, y, 
        width=0.7,      # 막대 너비
        color='blue')   # 막대 색상
plt.show()
```

### 조건부 색상 적용

```python
# 값이 7보다 크면 빨간색, 아니면 파란색
colors = ['red' if val > 7 else 'blue' for val in y]
plt.bar(x, y, width=0.7, color=colors)
```

**리스트 컴프리헨션 설명:**
```python
colors = []
for val in y:
    if val > 7:
        colors.append('red')
    else:
        colors.append('blue')
```

---

## 실전 예제

### 예제 1: 완성된 선 그래프

```python
import matplotlib.pyplot as plt

# 제목 설정
plt.title("Graph Title", fontsize=20, color='green', 
          fontweight='bold', loc='left', pad=10)

# 두 개의 선 그래프
plt.plot([1,2,3], [4,5,6], c='green', linewidth=2, 
         linestyle=':', label='Mouse')
plt.plot([1,2,3], [1,4,9], c='blue', linewidth=2, 
         linestyle=':', label='Cat')

# 격자선, 축 레이블, 범례
plt.grid(linestyle='--')
plt.xlabel('Sequence', fontsize=10, color='green', fontweight='bold')
plt.ylabel('Time(secs)', fontsize=10, color='green', fontweight='bold')
plt.legend(loc='upper left', fontsize=10)

# 축 범위
plt.xlim([0, 4])
plt.ylim([0, 10])

plt.show()
```

### 예제 2: 조건부 색상 막대 그래프

```python
import matplotlib.pyplot as plt

y = [5, 3, 7, 10, 9, 5, 3.5, 8]
x = range(len(y))

plt.title("Bar Graph", fontsize=20, fontweight='bold', 
          loc='left', pad=10)

# 조건부 색상: 7 초과는 빨간색, 이하는 파란색
colors = ['red' if val > 7 else 'blue' for val in y]
plt.bar(x, y, width=0.7, color=colors)

plt.show()
```

### 예제 3: 여러 그래프를 한 번에 표시

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))  # 전체 그림 크기 설정

# 첫 번째 그래프 (1행 2열 중 첫 번째)
plt.subplot(1, 2, 1)
plt.title("Line Graph")
plt.plot([1,2,3], [4,5,6], 'g:', label='Mouse')
plt.plot([1,2,3], [1,4,9], 'b:', label='Cat')
plt.legend()
plt.grid(True)

# 두 번째 그래프 (1행 2열 중 두 번째)
plt.subplot(1, 2, 2)
plt.title("Bar Graph")
y = [5, 3, 7, 10, 9, 5, 3.5, 8]
colors = ['red' if val > 7 else 'blue' for val in y]
plt.bar(range(len(y)), y, color=colors)

plt.tight_layout()  # 레이아웃 자동 조정
plt.show()
```

---

## 📝 핵심 요약

### 필수 함수들

| 함수 | 설명 | 예시 |
|------|------|------|
| `plt.plot()` | 선 그래프 | `plt.plot(x, y, 'r-')` |
| `plt.bar()` | 막대 그래프 | `plt.bar(x, y, color='blue')` |
| `plt.title()` | 제목 | `plt.title("제목")` |
| `plt.xlabel()` | X축 레이블 | `plt.xlabel("X축")` |
| `plt.ylabel()` | Y축 레이블 | `plt.ylabel("Y축")` |
| `plt.legend()` | 범례 | `plt.legend(['A', 'B'])` |
| `plt.grid()` | 격자선 | `plt.grid(True)` |
| `plt.xlim()` | X축 범위 | `plt.xlim([0, 10])` |
| `plt.ylim()` | Y축 범위 | `plt.ylim([0, 100])` |
| `plt.show()` | 그래프 표시 | `plt.show()` |

### 스타일 파라미터

- **색상**: `c` 또는 `color`
- **선 두께**: `linewidth` 또는 `lw`
- **선 스타일**: `linestyle` 또는 `ls`
- **글자 크기**: `fontsize`
- **글자 굵기**: `fontweight` ('normal', 'bold')

---

## 💡 유용한 팁

1. **색상과 스타일 단축 표기**
   ```python
   plt.plot(x, y, 'r--')  # 빨간색 점선
   plt.plot(x, y, 'bo')   # 파란색 원 마커
   plt.plot(x, y, 'g^')   # 녹색 삼각형 마커
   ```

2. **여러 데이터 한 번에 플롯**
   ```python
   plt.plot(x, y1, 'r-', x, y2, 'b--', x, y3, 'g:')
   ```

3. **그래프 저장**
   ```python
   plt.savefig('graph.png', dpi=300, bbox_inches='tight')
   ```

4. **한글 폰트 설정** (한글 깨짐 방지)
   ```python
   plt.rc('font', family='Malgun Gothic')  # Windows
   plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
   ```

---

## 🎯 연습 문제

1. 서로 다른 색상과 스타일의 3개 선을 그려보세요.
2. 막대 그래프에서 각 막대마다 다른 색상을 적용해보세요.
3. 2x2 형태로 4개의 그래프를 한 화면에 배치해보세요.

Happy Coding! 🚀