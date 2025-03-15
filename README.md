# Machine Learning cheats

Status: Drafting
Audience: Personal
Writer: Kisae Lee
Type of content: Blog, Tech
Start Date: February 25, 2025

# Numpy

행렬연산 도구

---

### 텐서(Tensor)

**정의**: 수학에서는 1차원 배열은 벡터(Vector), 2차원 배열은 행렬(matrix)이라고 함. 이를 일반화한 것을 텐서(Tensor) 라고 함.

벡터의 경우 축의 갯수에 따라 1차원, 2차원 벡터라고 함. 텐서의 경우 벡터의 축의 갯수를 랭크(rank)라고 함(n 랭크 텐서).

### **브로드캐스트(Broadcast)**

**정의**: 형상이 다른 배열끼리 연산이 가능하도록 형상을 확대 후 연산해 주는 기능.

예시

```python
import numpy as np

A = np.array([1,2], [3,4])
B = np.array([10, 20]) 
print(A * B) # Broadcast B to array([10, 20], [10, 20])

# array([10, 40], [30, 80])
```

---

# Matplotlib

그래프 시각화 도구

---

단순 그래프 시각화:

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate data from 0 to 6 with 0.1 interval
x = np.arange(0, 6, 0.1)
# Generate sin value of x
y1 = np.sin(x)
y2 = np.cos(x)

# Draw graph
plt.plot(x, y1, label="sin")
# Draw cos graph with dashed line
plt.plot(x, y2, linestyle="--", label="cos")
plt.xlabel("x")
plt.ylabel("y")
plt.title("sin & cos")
plt.legend()
plt.show()
```

단순 이미지 시각화:

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

img = imread("assets/sample_image.jpeg")

plt.imshow(img)
plt.show()
```

---

# 퍼셉트론(Perceptron)

프랑크 로젠블라트(Frank Rosenblatt)가 1957년에 고안한 알고리즘. 신경망(딥러닝)의 기원이 되는 알고리즘.

---

### 정의

퍼셉트론은 다수의 신호를 입력으로 받아 하나의 신호를 출력하는 것.

```mermaid
flowchart LR
  a((x_1)) -- "w_1" --> b((y))
  c((x_2)) -- "w_2" --> b((y))
```

- $x_1$, $x_2$는 입력 신호, $y$는 출력 신호, $w_1$, $w_2$는 가중치를 뜻함. ($w$는 weight의 머리글자, 가중치는 전류에서 말하는 저항에 해당. 퍼셉트론의 가중치는 그 값이 클수록 강한 신호를 보냄.)
- 그림의 원: 뉴런 혹은 노드라고 칭함.
- 뉴런에서 보내온 신호의 총합이 정해진 한계(임계값, $\theta$)를 넘어설 때만 1을 출력. (뉴런이 활성화한다)
- 동작원리를 수식으로 표현하면 아래와 같다.

$$
y = \begin{cases} 0 (w_1 x_1 + w_2 x_2 <= \theta) \\ 1 (w_1 x_1 + w_2 x_2 > \theta) \end{cases}
$$

### 퍼셉트론과 논리회로

**AND 게이트**

진리표:

| $x_1$ | $x_2$ | $y$ |
| --- | --- | --- |
| 0 | 0 | 0 |
| 1 | 0 | 0 |
| 0 | 1 | 0 |
| 1 | 1 | 1 |

좌측 진리표대로 작동하도록 하는 $w_1$, $w_2$, $\theta$ 값은 (0.5, 0.5, 0.7), (0.5, 0.5, 0.8), (1.0, 1.0, 1.0) 때 모두 만족한다.

**NAND 게이트**

진리표:

| $x_1$ | $x_2$ | $y$ |
| --- | --- | --- |
| 0 | 0 | 1 |
| 1 | 0 | 1 |
| 0 | 1 | 1 |
| 1 | 1 | 0 |

좌측 진리표대로 작동하도록 하는 $w_1$, $w_2$, $\theta$ 값은 AND 게이트의 $w_1$, $w_2$, $\theta$값을 반전하면 된다.

**OR 게이트**

진리표:

| $x_1$ | $x_2$ | $y$ |
| --- | --- | --- |
| 0 | 0 | 0 |
| 1 | 0 | 1 |
| 0 | 1 | 1 |
| 1 | 1 | 1 |

좌측 진리표대로 작동하도록 하는 $w_1$, $w_2$, $\theta$ 값은 (0.5, 0.5, -0.7), (0.5, 0.5, -0.8), (1.0, 1.0, 0.0) 때 모두 만족한다.

기존 임계값 $\theta$ 를 $-b$로 치환한 공식은 아래와 같다. 이 때 $b$ 를 편향(bias)라고 한다.

$$
y = \begin{cases} 0 (b + w_1 x_1 + w_2 x_2 <= 0) \\ 1 (b + w_1 x_1 + w_2 x_2 > 0) \end{cases}
$$

편향, 가중치 공식을 활용하여 AND, NAND, OR 퍼셉트론 구현은 아래와 같다.

```python
import numpy as np

def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1
```

**XOR 게이트**

진리표:

| $x_1$ | $x_2$ | $y$ |
| --- | --- | --- |
| 0 | 0 | 0 |
| 1 | 0 | 1 |
| 0 | 1 | 1 |
| 1 | 1 | 0 |

XOR 게이트는 퍼셉트론으로 구현할 수 없다. XOR 게이트는 비선형 영역으로 이루어져 있기 때문이다.

### 다층 퍼셉트론(multi-layer perceptron)

XOR 게이트는 AND, NAND, OR 게이트를 조합하여 만들 수 있다.

```mermaid
graph LR
  x1((x_1)) --> s1((s_1 - NAND))
  x1 --> s2((s_2 - OR))
  x2((x_2)) --> s1
  x2 --> s2
  s1 --> AND((AND))
  s2 --> AND
  AND --> y((y))
```

위 조합을 기준으로 XOR 진리표를 다시 작성하면 아래와 같다.

| $x_1$ | $x_2$ | $s_1$ | $s_2$ | $y$ |
| --- | --- | --- | --- | --- |
| 0 | 0 | 1 | 0 | 0 |
| 1 | 0 | 1 | 1 | 1 |
| 0 | 1 | 1 | 1 | 1 |
| 1 | 1 | 0 | 1 | 0 |

XOR 게이트의 구현은 아래와 같다.

```python
def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y
```

즉 XOR 은 다층 구조의 네트워크이며, 2층의 퍼셉트론(다층 퍼셉트론)이라는 것을 알 수 있다.

```mermaid
graph LR
  x1((x_1)) --> s1((s_1))
  x1 --> s2((s_2))
  x2((x_2)) --> s1
  x2 --> s2
  s1 --> y((y))
  s2 --> y

```

- 0 층: $x_1$, $x_2$
- 1 층: $s_1$, $s_2$
- 2 층: $y$

과정에 대한 상세 기술:

<aside>

1. 0 층 뉴런이 입력 신호를 받아 1층의 뉴런으로 신호를 보낸다.
2. 1 층 뉴런이 2층의 뉴런으로 신호를 보내고, 2층의 뉴런은 $y$를 출력한다.

작업자들 사이에서 **부품을 전달**하는 일이 이뤄진다.

</aside>

**Key Point:** 

- 단층 퍼셉트론으로는 표현하지 못한 것을 층을 늘려 구현할 수 있다.
- 다층 퍼셉트론은 이론상 컴퓨터를 표현할 수 있다.

# 신경망

신경망은 입력층, 출력층, 은닉층으로 구성된다. 은닉층은 사람 눈에 보이지 않는다.

---

### 퍼셉트론에서 신경망으로

```mermaid
graph LR
  x1((x_1)) --w_1--> y
  x2((x_2)) --w_2--> y((y))

```

두 신호를 받아 $y$를 출력하는 퍼셉트론. 공식은 아래와 같다.

$$
y = \begin{cases} 0 (b + w_1 x_1 + w_2 x_2 <= 0) \\ 1 (b + w_1 x_1 + w_2 x_2 > 0) \end{cases}
$$

- **편향**: 뉴런이 얼마나 쉽게 활성화되는냐를 제어
- **가중치**: 각 신호의 영향력을 제어

편향을 명시하여 표시한다면 아래와 같다. (편향의 입력 신호는 항상 1 이다)

```mermaid
graph LR
  1((1)) --$$b$$--> y
  x1((x_1)) --w_1--> y
  x2((x_2)) --w_2--> y((y))
  style 1 fill:#aaaa

```

더 간결한 형태로 식을 작성하면 아래와 같다.

$$
y = h(b + w_1 x_1 + w_2 x_2)
$$

$$
h(x) = \begin{cases} 0 (x<=0) \\ 1 (x > 0) \end{cases}
$$

위 $h(x)$라는 함수는 입력 신호의 총합을 출력 신호로 변환하는 역할을 하며, 일반적으로 이를 **활성함수(activation function)**라 한다.  처리 과정을 세분화해서 표현한 것은 아래와 같다.

$$
\begin{align} a = b + w_1 x_1 + w_2 x_2 \\ y = h(a) \end{align}
$$

```mermaid
graph LR
  1((1)) --b--> a
  x1((x_1)) --w_1--> a
  x2((x_2)) --w_2--> a
  
  subgraph "h()"
  a((a)) --> y((y))
  end
  
  style 1 fill:#aaaa

```

### 활성화 함수

아래 식과 같은 활성화 함수는 임계값을 경계로 출력이 바뀜. 이를 **계단 함수**(step function)라 함.

$$
h(x) = \begin{cases} 0 (x<=0) \\ 1 (x > 0) \end{cases}
$$

계단 함수 외 다른 함수로 변경하는 것이 신경망으로 나아가는 열쇠. 신경망에서 자주 이용하는 활성화 함수 - **시그모이드 함수**(sigmoid function)

$$
h(x) = \frac{1}{1 + exp(-x)}
$$

식에서 $exp(-x)$ 은 $e^{-x}$ 를 뜻함. $e$는 자연상수로 2.7182… 값을 갖는 실수.

퍼셉트론과 신경망의 주된 차이는 활성화 함수 뿐.

### 계단 함수 구현

단순한 계단 함수:

```python
import numpy as np

def step_function(x):  # Only takes a number
    if x > 0:
        return 1
    else:
        return

def step_function_array(x):  # It can takes an array
    y = x > 0  # It will compare each element of the array with 0
    return y.astype(np.int_)  # It will convert the boolean array to int array

print(step_function(-1))  # 0
print(step_function_array(np.array([-1.0, 2.0, 3.0])))  # [0 1 1]
```

`astype(dtype)` 함수: 대상 배열 중 모든 요소를 입력한 데이터 타입으로 변환한다.

### 시그모이드 함수 구현

```python
import numpy as np
import matplotlib.pylab as plt

def step_function_array(x):  # It can takes an array
    y = x > 0  # It will compare each element of the array with 0
    return y.astype(np.int_)  # It will convert the boolean array to int array

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x1 = np.arange(-5.0, 5.0, 0)
y1 = step_function_array(x1)

x2 = np.arange(-5.0, 5.0, 0.1)
y2 = sigmoid(x2)

plt.plot(x1, y1, linestyle="--", label="step")
plt.plot(x2, y2, label="sigmoid")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.ylim(-0.1, 1.1)
plt.show()
```

계단 함수와 시그모이드 함수 모두 출력은 0~1 사이라는 것은 동일하다.

### 비선형 함수

계단 함수와 시그모이드 함수 모두 비선형 함수이다. 선형함수는 무언가를 입력했을 때 출력이 입력의 상수배만큼 변하는 함수이다. $f(x) = ax + b$ 가 대표적인 선형함수이며, 이때 $a$ 와 $b$ 는 상수다.

<aside>

신경망에서 비선형 함수를 사용하는 이유:

선형함수를 활성화 함수로 채택한 경우의 예: 

- 선형함수: $h(x)=cx$
- 3층 네트워크를 위 선형함수로 대입한 결과: $y(x)=h(h(h(x)))$
- 변환 후: $y(x)=c*c*c*x$
- 이것은 $y(x)=ax$ 와  본질적으로 같음($a=c^3$이라고 한다면).
- 즉 은닉층이 없는 네트워크
</aside>

### ReLU 함수

최근에는 ReLU(Rectified Linear Unit, 렐루) 함수를 주로 이용.

```python
import numpy as np

def relu(x):
    return np.maximum(0, x)
    
x1 = np.arange(-5.0, 5.0, 0.1)
y1 = step_function_array(x1)

x2 = np.arange(-5.0, 5.0, 0.1)
y2 = sigmoid(x2)

x3 = np.arange(-5.0, 5.0, 0.1)
y3 = relu(x3)

plt.plot(x1, y1, linestyle="--", label="step")
plt.plot(x2, y2, label="sigmoid")
plt.plot(x3, y3, label="ReLU")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
```

$$
h(x) = \begin{cases} 0 (x<=0) \\ x (x > 0) \end{cases}
$$

### 다차원 배열

차원 수 및 다차원 배열의 모양 확인하기

```python
import numpy as np

A = np.array([[1, 2, 3, 4]])
print(np.ndim(A))  # 1. Returns the number of dimensions of the array
print(A.shape)  # (4,). Returns the shape of the array

B = np.array([[1, 2], [3, 4], [5, 6]])
print(np.ndim(B))  # 2
print(B.shape)  # (3, 2)
```

행렬 곱 계산하기 (`np.dot()`)

- `np.dot(A, B)` 와 `np.dot(B, A)` 는 다른 값이 될 수 있다. (행렬곱 공식확인)
- 행렬 곱은 행렬 A 의 1 번째 차원의 원소 수와 행렬 B 의 0번째 차원의 원소 수가 같아야 함.
- 즉, 행렬 A 의 형상이 $(row_A, column_A)$ , 행렬 B 의 형상이 $(row_B, column_B)$ 일 때, $column_A = row_B$ 가 만족되어야 한다.
- 행렬 곱의 결과 C 의 형상은 $(row_A, column_B)$ 가 된다.

```python
A = np.array([[1, 2, 3], [4, 5, 6]])
B = np.array([[1, 2], [3, 4], [5, 6]])
C = np.dot(A, B)
print(C)  # [[22 28] [49 64]]
```

신경망에서의 행렬 곱

```mermaid
graph LR
  x1((x_1)) --1--> y1
  x2((x_2)) --2--> y1((y_1))
  x1((x_1)) --3--> y2
  x2((x_2)) --4--> y2((y_2))
  x1((x_1)) --5--> y3
  x2((x_2)) --6--> y3((y_3))

```

위와 같은 신경망 형태에서, X 는 (2,), W 는(2,3), Y는(,3) 형상을 갖는다. 편향과 활성화 함수를 생략하고 가중치만 고려했을 때, Y 값을 계산하는 방법은 아래와 같다.

```python
# In neural network
X = np.array([1, 2])
W = np.array([[1, 3, 5], [2, 4, 6]])
Y = np.dot(X, W)
print(Y)  # [5 11 17]
```

즉 행렬곱을 통해 Y 는 한 번의 연산으로 계산가능하다.

## 신경망 구현하기

```mermaid
graph LR
  x2((x_2)) --w_12--> a1((a_1))

```

$$
w^{(1)}_{12}
$$

- $(1)$: 1 층에 가중치라는 의미
- $1$: 다음 층 뉴런의 번호 (다음 층 1 번째 뉴런)
- $2$: 앞 층 뉴런의 번호 (앞 층 2 번째 뉴런)

편향을 포함했을 때의 관계도는 아래와 같음($a_2$, $a_3$ 는 귀찮아서 그리지 않음)

```mermaid
graph LR
  1((1)) --b_1--> a((a_1))
  x1(($$x_1$$)) --w_11--> a
  x2(($$x_2$$)) --w_12--> a
  
  style 1 fill:#aaaa

```

$a^{(1)}_{1}$의 값을 계산하는 공식:

$$
a^{(1)}_{1} = w^{(1)}_{11}x_1 + w^{(1)}_{12}x2 + b^{(1)}_{1}
$$

행렬 곱으로 전환 후,

$$
A^{(1)} = XW^{(1)} + B(1)
$$

이때, 각 행렬은 아래와 같음

$$
A^{(1)} = (a^{(1)}_{1}, a^{(1)}_{2}, a^{(1)}_{3}), X = (x_1, x_2), B^{(1)} = (b^{(1)}_{1}, b^{(1)}_{2}, b^{(1)}_{3})
$$

$$
W^{(1)} = \begin{pmatrix} w^{(1)}_{11} & w^{(1)}_{21} & w^{(1)}_{31}\\ w^{(1)}_{12} & w^{(1)}_{22} & w^{(1)}_{32}\end{pmatrix}
$$

이것을 절차지향적으로 구현한 결과는 아래과 같음

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# this function is not doing anything
# in this specific case
def identity_function(x):
    return x

# input layer to level 1
X = np.array([1.0, 0.5])
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])

A1 = np.dot(X, W1) + B1
Z1 = sigmoid(A1)
print(Z1)  # [0.57444252 0.66818777 0.75026011]

# level 1 to level 2
W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])

A2 = np.dot(Z1, W2) + B2
Z2 = sigmoid(A2)
print(Z2)  # [0.62624937 0.7710107]

# level 2 to output layer
W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([0.1, 0.2])

A3 = np.dot(Z2, W3) + B3
Y = identity_function(A3)
print(Y)  # [0.31682708 0.69627909]
```

위 구현을 정리하면 아래와 같음

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# this function is not doing anything
# in this specific case
def identity_function(x):
    return x

def init_network():  # return a network with 2 inputs
    network = {}
    network["W1"] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network["b1"] = np.array([0.1, 0.2, 0.3])
    network["W2"] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network["b2"] = np.array([0.1, 0.2])
    network["W3"] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network["b3"] = np.array([0.1, 0.2])

    return network

def forward(network, x):
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)

    return y

network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y)  # [0.31682708 0.69627909]
```

## 출력층 설계하기

기계학습 문제는 **분류**(classification)와 **회귀**(regression)로 나뉨. 사진 속 인물의 성별을 분류하는 문제는 ‘분류’, 사진 속 인물의 몸무게를 예측하는 문제는 ‘회귀’.

일반적으로 ‘회귀’에는 **항등 함수**(identify function)을 사용. 항등 함수는 입력을 그대로 출력함. ‘항등’은 입력과 출력이 항상 같다는 의미.

‘분류’에는 **소프트맥스 함수**(softmax function) 을 사용. 식은 다음과 같음.

$$
y_k=\frac{exp(a_k)}{\sum_{i=1}^n exp(a_i)}
$$

소프트맥스 함수의 구현은 다음과 같음.

```python
import numpy as np

def softmax(a):  # this will overflow
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y
```

다만, 소프트맥스 함수는 지수함수를 사용하여, 엄청 큰 값을 리턴할 수 있음. 또한 이러한 큰 값끼리 나눗셈을 할 경우 결과 수치가 불안정해짐. 이 때, 소프트맥스 함수를 아래와 같이 개선하여 구현하면 오버플로 문제를 해결할 수 있음.

$$
y_k=\frac{exp(a_k-C')}{\sum_{i=1}^n exp(a_i-C')}
$$

그 구현은 아래와 같음.

```python
import numpy as np

def softmax_modified(a):  # this will not overflow
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y
```

- 소프트맥스 함수의 성질: 출력은 0.0~1.0 사이 실수이며, 출력의 총 합은 1. 즉, 소프트맥스 함수의 출력을 ‘확률’로 해석할 수 있음.
- ‘분류’ 문제에서, 은닉층의 마지막 층 원소의 대소 관계와, 소프트맥스 함수를 통해 계산된 출력층 원소의 대소 관계는 동일함으로 굳이 소프트맥스 함수를 통해 추가 계산하지 않음.

## 손글씨 숫자 인식

학습과정은 생략하고, 추론과정만 구현. 이 추론 과정을 신경망의 **순전파**(forward propagation) 이라고 함.

<aside>

MNIST 데이터 가져오기:

- tensorflow 를 통한 MNIST 가져오기 (딱봐도 이게 좋다)

```python
# pip install tensorflow
# Better! load the MNIST dataset using tensorflow
import keras

import keras
import matplotlib.pyplot as plt

Mnist = keras.datasets.mnist
(x_train, t_train), (x_test, t_test) = Mnist.load_data()

print(x_train.shape, t_train.shape)
print(x_test.shape, t_test.shape)

for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.tight_layout()
    plt.imshow(x_train[i].reshape(28, 28), cmap="gray", interpolation="none")
    plt.title("digit: {}".format(t_train[i]))
    plt.xticks([])
    plt.yticks([])

plt.show()
```

- pytorch 를 통한 MNIST 가져오기

```python
# pip install torch
# pip install torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

# Load the MNIST dataset using torchvision.datasets
BATCH_SIZE = 32
train_set = datasets.MNIST(
    root="./mnist", train=True, transform=ToTensor(), download=True
)

test_set = datasets.MNIST(
    root="./mnist", train=False, transform=ToTensor(), download=True
)

print(len(train_set))

train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=BATCH_SIZE, shuffle=False)

for x_train, t_train in train_loader:
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.tight_layout()
        plt.imshow(x_train[i].reshape(28, 28), cmap="gray", interpolation="none")
        plt.title("digit: {}".format(t_train[i]))
        plt.xticks([])
        plt.yticks([])

    plt.show()
    break
```

</aside>

### 신경망 추론 처리

```python
import keras
import numpy as np
import pickle

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

def get_data():
    Mnist = keras.datasets.mnist
    (x_train, t_train), (x_test, t_test) = Mnist.load_data()
    # Normalize the image data
    return x_test.reshape([-1, 28 * 28]) / 255, t_test

def init_network():
    with open("assets/sample_weight.pkl", "rb") as f:
        network = pickle.load(f)

    return network

def predict(network, x):
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y

x, t = get_data()
network = init_network()

accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i].flatten())
    p = np.argmax(y)
    if p == t[i]:
        accuracy_cnt += 1

print("Accuracy: " + str(float(accuracy_cnt) / len(x)))
```

- 정확도: 0.9352
- 여기서 0 ~ 255 범위인 픽셀의 값을 0.0 ~ 1.0 범위로 변환했다(안하면 시그모이드 함수에서 오버플로가 난다). 이러한 변환을 **정규화**(normalization)라고 한다.
- 이렇게 신경망의 입력 데이터 특정 변환을 가하는 것을 **전처리**(pre-processing)라 한다. ‘정규화’는 ‘전처리’ 작업 중 하나이다.

### 배치처리(묶음처리)

```python
batch_size = 100
for i in range(0, len(x), batch_size):
    x_batch = x[i : i + batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p == t[i : i + batch_size])

print("Accuracy: " + str(float(accuracy_cnt) / len(x)))
```

# 신경망 학습

선형 분리 가능 문제라면 데이터로부터 자동으로 학습할 수 있지만, 비선형 분리 문제는 자동으로 학습할 수 없다. 선형 분리 가능 문제는 유한 번의 학습을 통해 풀 수 있다는 사실이 **퍼셉트론 수렴 정리**(perceptron convergence theorem)으로 증명 됨.

## 데이터 주도 학습

- 전통적인 방식으로서, **사람이 데이터를 확인하고 알고리즘을 개발**하여 결과를 도출한다.
- **기계학습** 기술 중:
    - 이미지에서 ‘**특징**(feature)’을 추출하고 그 특징의 패턴을 학습하는 방법
        - 이미지 특징은 보통 벡터로 기술한다.
        - CV 분야에서 SIFT, SURF, HOG 등 특징을 많이 사용한다.
        - 변환된 벡터를 가지고 지도 학습 방식의 대표 분류 기법인 SVM, KNN 등을 통해 학습할 수 있다.
- **신경망** 방식은 전 과정에서 사람이 개입하지 않는다.
- **딥러닝**을 **종단간 기계학습**(end-to-end machine learning)이라고도 한다.

회색 블록은 사람이 개입하지 않음을 뜻함

```mermaid
graph LR
  1[데이터] --> 2[사람이 생각한 알고리즘]
  2 --> 3[결과]
  4[데이터] --> 5["사람이 생각한 특징(SIFT, HOG 등)"]
  5 --> 6["기계학습(SVN, KNN 등)"]
  6 --> 7[결과]
  8[데이터] --> 9["신경망(딥러닝)"]
  9 --> 10[결과]
  
  style 6 fill:#aaaa
  style 9 fill:#aaaa
  
```

기계학습 문제에서의 데이터

- **훈련 데이터**(trainning data)와 **시험 데이터**(test data)가 존재한다.
- 시험 데이터는 **범용 능력**을 제대로 평가하기 위한 데이터이다.
- 하나의 데이터셋에만 지나치게 최적화된 상태를 **오버피팅**(overfitting)이라고 한다.

## 손실함수

**손실함수**(loss function)는 신경망 성능의 ‘나쁨’을 나타내는 지표로, 훈련 데이터를 얼마나 잘 처리하지 ‘못’하느냐를 나타냄. 일반적으로는 오차제곱합과 교차 엔트로피 오차를 사용.

**오차제곱합**(sum of squares for error, SSE):

$$
E=\frac{1}{2}\sum_{k} (y_k-t_k)^2
$$

- $y_k$: 신경망의 출력
- $t_k$: 정답 레이블
- $k$: 데이터의 차원 수

예시(한 원소만 1로 하고, 그 외는 0으로 나타내는 표기법을 **원-핫 인코딩**(one-hot encoding)이라 함):

```bash
>>> y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0] # 신경망의 출력. 소프트맥스 함수 활용.
>>> t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] # 정답 레이블
```

구현:

```python
import numpy as np

def sum_squares_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)

y = [
    0.1,
    0.05,
    0.6,
    0.0,
    0.05,
    0.1,
    0.0,
    0.1,
    0.0,
    0.0,
]  # output of the neural network. Estimated most likely '2'. 
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  # answer label

print(sum_squares_error(np.array(y), np.array(t)))  # 0.09750000000000003

```

**교차 엔트로피 오차**(cross entropy error, CEE)

$$
E=-\sum_{k}t_k\log_e{y_k}
$$

구현:

```python
import numpy as np

def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))

y = [
    0.1,
    0.05,
    0.6,
    0.0,
    0.05,
    0.1,
    0.0,
    0.1,
    0.0,
    0.0,
]  # output of the neural network
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  # answer label

print(cross_entropy_error(np.array(y), np.array(t)))  # 0.510825457099338
```

모든 훈련데이터를 대상으로 손실 함수 값을 구하기(**미니배치**(mini-batch)):

- 일반적으로, 모든 데이터를 대상으로 손실 함수의 합을 구하는 것은 많은 시간이 소요된다. 고로 데이터 일부를 추려 전체의 ‘근사치’로 이용한다. 신경망 학습에서도 훈련 데이터로부터 일부만 골라 학습을 수행. 이 일부를 **미니배치**라고 한다.
- 교차 엔트로피 오차를 활용한 배치 → ‘평균 손실 함수’를 구하는 것

$$
E=-\frac{1}{N}\sum{n}\sum_{k}t_{nk}\log_e{y_{nk}}
$$

- $N$: 총 데이터 수
- $t_{nk}$: $n$번째 데이터의 $k$번째 값

<aside>

신경망을 학습할 떄 정확도를 지표로 삼아서는 안 된다. 정확도를 지표로 하면 매개변수의 미분이 대부분의 장소에서 0이 되기 때문이다.

즉, 정확도를 지표로 한다면, 매개변수 조정을 통해 정확도가 개선이 되지 않는 상황이나, 개선될지라도 정확도가 연속적인 변화를 띄지 않는 경우가 존재한다. 

</aside>

## 수치 미분(numerical differentiation)

- 미분은 ‘특정 순간’의 변화량을 뜻한다.
- 수치 미분은 아죽 작은 차분으로 미분하는 것.
- 수식을 전개해 미분하는 것은 ‘해석적(analytic)’이라는 말을 사용.

$$
\frac{df(x)}{dx} = \lim\nolimits_{h \to 0}{\frac{f(x+h) - f(x)}{h}}
$$

수치 미분은 컴퓨터에서 구현 시 아래와 같은 이슈가 있을 수 있다.

- **반올림 오차**(rounding error): $1e-50$을 float32형으로 나타내면 0.0이 되어, 올바로 표현 불가. $1e-4$정도에서 좋은 결과를 얻을 수 있음.
- 근사로 구한 접선이기 때문에 엄밀한 일치가 어려움.

오차를 줄이는 방법:

- $(x+h)$와 $(x-h)$일 때의 함수$f$ 의 차분을 계산하는 방법 → **중심 차분** 혹은 **중앙 차분**

구현:

```python
def numerical_diff(f, x):
	h = 1e-4 # 0.0001
	return (f(x+h) - f(x-h)) / (2*h)
```

수치 미분 예시:

$$
y = 0.01x^2+0.1x
$$

수치미분 구현:

```python
import numpy as np
import matplotlib.pyplot as plt

def function_1(x):
    return 0.01 * x**2 + 0.1 * x

def numerical_diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)

def tangent_line(f, x):
    d = numerical_diff(f, x)
    y = f(x) - d * x
    return lambda t: d * t + y

x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)

plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x, y)

plt.scatter(5, function_1(5))

tf = tangent_line(function_1, 5)
y2 = tf(x)
plt.plot(x, y2)

plt.show()
```

## 편미분(partial differentiation)

예시:

$$
f(x_0, x_1) = x_0^2+x_1^2
$$

**기울기**(gradient, 편미분을 벡터로 정리한 것) 구하기: 

$$
(\frac{\partial f}{\partial x_0}, \frac{\partial f}{\partial x_1})
$$

구현(이해하는데 난이도 있음):

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)  # generate array with same shape as x

    for idx in range(x.size):
        tmp_val = x[idx]
        # f(x+h)
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # f(x-h)
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val  # restore value

    return grad

# input is 1D array with 2 elements
def function_2(x):
    # return x[0] ** 2 + x[1] ** 2
    return np.sum(x**2)

x = np.linspace(-3, 3, 25)
y = np.linspace(-3, 3, 25)
X, Y = np.meshgrid(x, y)

Z = np.array(
    [
        [function_2(np.array([xi, yi])) for xi, yi in zip(row_x, row_y)]
        for row_x, row_y in zip(X, Y)
    ]
)

grad = np.array(
    [
        [
            numerical_gradient(function_2, np.array([xi, yi]))
            for xi, yi in zip(row_x, row_y)
        ]
        for row_x, row_y in zip(X, Y)
    ]
)

print(grad.shape)

# Extract gradient components
U = grad[:, :, 0]  # Gradient in x direction
V = grad[:, :, 1]  # Gradient in y direction

print(U.shape, V.shape)
print(U)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(X, Y, Z, cmap="viridis")

plt.show()

fig = plt.figure()
# Quiver graph
plt.quiver(X, Y, U, V)

plt.show()
```

## 경사법(경사 하강법)

매개변수 공간이 광대하여 어디가 최솟값이 되는 곳인지를 짐작하기 어려움. 이런 상황에서 기울기를 이용해 함수의 최솟값을 찾으려하는 것이 **경사법**(gradient method).

- 함수의 극솟값, 최솟값, 또 **안장점**(addle point)이 되는 장소에서는 기울기가 0
- 안장점은 어느 방향에서 보면 극댓값이고 다른 방향에서 보면 극솟값이 되는 점
- 경사법은 기울기가 0인 장소를 찾지만 그것이 반드시 최솟값이라고 할 수 없음
- 복잡하고 찌그러진 모양의 함수라면 평평한 곳으로 파고들면서 **고원**(plateau, 플래토)이라 하는, 학습이 진행되지 않는 정체기에 빠질 수 있음

경사법의 수식 표현:

$$
x_0 = x_0 - \eta\frac{\partial f}{\partial x_0}
$$

$$
x_1 = x_1 - \eta\frac{\partial f}{\partial x_1}
$$

- $\eta$ 기호(eta, 에타)는 갱신하는 양을 나타 냄.
- 이를 신경망 학습에서는 **학습률**(learning rate)이라고 함. → 즉 매개변수 값을 얼마나 갱신하느냐를 정하는 것
- 학습률 값은 특정 값으로 정해두어야 함. 일반적으로 너무 크거나 작으면 ‘좋은 장소’를 찾을 수 없음.
- 학습률 값을 변경하면서 올바르게 학습하고 있는지를 확인

구현:

```python
import numpy as np
import matplotlib.pyplot as plt

def function_2(x):
    # return x[0] ** 2 + x[1] ** 2
    return np.sum(x**2)

# Consider add type hints for numpy arrays
# reference: https://stackoverflow.com/questions/71109838/numpy-typing-with-specific-shape-and-datatype
def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]

        x[idx] = tmp_val + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val

    return grad

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    """
    init_x: initial value. 1D array.
    """
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x

def gradient_descent_recording(f, init_x, lr=0.01, step_num=100):
    """
    init_x: initial value. 1D array.
    """
    record = []
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
        record.append(x.copy())

    return np.array(record)

init_x = np.array([-3.0, 4.0])
record = gradient_descent_recording(function_2, init_x, lr=0.1, step_num=100)

fig = plt.figure()
plt.scatter(record[:, 0], record[:, 1])
plt.xlim(-3, 3)
plt.ylim(-4, 4)
plt.xlabel("X0")
plt.ylabel("X1")
plt.show()
```

학습률 같은 매개변수를 하이퍼파라미터(hyper parameter)라고 함.

- 직접 설정. 여러 후보 값 중에서 시험을 통해 가장 잘 학습하는 값을 찾아야 함

가중치, 편향 같은 신경망의 매개변수와는 성질이 다름.

- 가중치, 편향 → 훈련 데이터와 학습 알고리즘에 의해서 ‘자동’ 획득

## 학습 알고리즘 구현

<aside>

확률적 경사 하강법(stochastic gradient descent, SGD) → 미니배치를 무작위로 선정하기 때문

전제:

- 신경망에 적응 가능한 가중치와 편향이 있음.

1 단계 - 미니배치:

- 훈련 데이터 중 일부를 무작위로 가져 옴.

2 단계 - 기울기 산출:

- 미니배치의 손실 함수 값을 줄이기 위해 각 가중치 매개변수의 기울기를 구함

3 단계 - 매개변수 갱신:

- 가중치 매개변수를 기울기 방향으로 아주 조금 갱신

4 단계 - 반복:

- 1~3단계를 반복
</aside>

구현:

```python
from matplotlib import pyplot as plt
import numpy as np
import keras
from keras import utils

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))

def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    # TODO: Need to test it with a simple function
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]

        x[idx] = float(tmp_val) + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val  # restore value
        it.iternext()

    return grad

def get_data():
    Mnist = keras.datasets.mnist
    (x_train, t_train), (x_test, t_test) = Mnist.load_data()
    # Normalize and one-hot the image data
    return (
        x_train.reshape([-1, 28 * 28]) / 255,
        utils.to_categorical(t_train),
        x_test.reshape([-1, 28 * 28]) / 255,
        utils.to_categorical(t_test),
    )

class two_layer_net:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params["W1"] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params["b1"] = np.zeros(hidden_size)
        self.params["W2"] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params["b2"] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params["W1"], self.params["W2"]
        b1, b2 = self.params["b1"], self.params["b2"]

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y

    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads["W1"] = numerical_gradient(loss_W, self.params["W1"])
        grads["b1"] = numerical_gradient(loss_W, self.params["b1"])
        grads["W2"] = numerical_gradient(loss_W, self.params["W2"])
        grads["b2"] = numerical_gradient(loss_W, self.params["b2"])

        return grads

train_loss_list = []
train_acc_list = []
test_acc_list = []

iters_num = 10000
batch_size = 100
learning_rate = 0.1

x_train, t_train, x_test, t_test = get_data()
train_size = x_train.shape[0]

iter_per_epoch = max(train_size / batch_size, 1)

net = two_layer_net(input_size=784, hidden_size=50, output_size=10)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grad = net.numerical_gradient(x_batch, t_batch)

    for key in ("W1", "b1", "W2", "b2"):
        net.params[key] -= learning_rate * grad[key]

    loss = net.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    print(f"iteration {i}: loss {loss}")

    if i % iter_per_epoch == 0:
        train_acc = net.accuracy(x_train, t_train)
        test_acc = net.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(f"train acc, test acc | {train_acc}, {test_acc}")

plt.plot(np.arange(iters_num), train_loss_list)
plt.show()

```

# 신경망 학습

선형 분리 가능 문제라면 데이터로부터 자동으로 학습할 수 있지만, 비선형 분리 문제는 자동으로 학습할 수 없다. 선형 분리 가능 문제는 유한 번의 학습을 통해 풀 수 있다는 사실이 **퍼셉트론 수렴 정리**(perceptron convergence theorem)으로 증명 됨.

## 데이터 주도 학습

- 전통적인 방식으로서, **사람이 데이터를 확인하고 알고리즘을 개발**하여 결과를 도출한다.
- **기계학습** 기술 중:
    - 이미지에서 ‘**특징**(feature)’을 추출하고 그 특징의 패턴을 학습하는 방법
        - 이미지 특징은 보통 벡터로 기술한다.
        - CV 분야에서 SIFT, SURF, HOG 등 특징을 많이 사용한다.
        - 변환된 벡터를 가지고 지도 학습 방식의 대표 분류 기법인 SVM, KNN 등을 통해 학습할 수 있다.
- **신경망** 방식은 전 과정에서 사람이 개입하지 않는다.
- **딥러닝**을 **종단간 기계학습**(end-to-end machine learning)이라고도 한다.

회색 블록은 사람이 개입하지 않음을 뜻함

```mermaid
graph LR
  1[데이터] --> 2[사람이 생각한 알고리즘]
  2 --> 3[결과]
  4[데이터] --> 5["사람이 생각한 특징(SIFT, HOG 등)"]
  5 --> 6["기계학습(SVN, KNN 등)"]
  6 --> 7[결과]
  8[데이터] --> 9["신경망(딥러닝)"]
  9 --> 10[결과]
  
  style 6 fill:#aaaa
  style 9 fill:#aaaa
  
```

기계학습 문제에서의 데이터

- **훈련 데이터**(trainning data)와 **시험 데이터**(test data)가 존재한다.
- 시험 데이터는 **범용 능력**을 제대로 평가하기 위한 데이터이다.
- 하나의 데이터셋에만 지나치게 최적화된 상태를 **오버피팅**(overfitting)이라고 한다.

## 손실함수

**손실함수**(loss function)는 신경망 성능의 ‘나쁨’을 나타내는 지표로, 훈련 데이터를 얼마나 잘 처리하지 ‘못’하느냐를 나타냄. 일반적으로는 오차제곱합과 교차 엔트로피 오차를 사용.

**오차제곱합**(sum of squares for error, SSE):

$$
E=\frac{1}{2}\sum_{k} (y_k-t_k)^2
$$

- $y_k$: 신경망의 출력
- $t_k$: 정답 레이블
- $k$: 데이터의 차원 수

예시(한 원소만 1로 하고, 그 외는 0으로 나타내는 표기법을 **원-핫 인코딩**(one-hot encoding)이라 함):

```bash
>>> y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0] # 신경망의 출력. 소프트맥스 함수 활용.
>>> t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] # 정답 레이블
```

구현:

```python
import numpy as np

def sum_squares_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)

y = [
    0.1,
    0.05,
    0.6,
    0.0,
    0.05,
    0.1,
    0.0,
    0.1,
    0.0,
    0.0,
]  # output of the neural network. Estimated most likely '2'. 
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  # answer label

print(sum_squares_error(np.array(y), np.array(t)))  # 0.09750000000000003

```

**교차 엔트로피 오차**(cross entropy error, CEE)

$$
E=-\sum_{k}t_k\log_e{y_k}
$$

구현:

```python
import numpy as np

def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))

y = [
    0.1,
    0.05,
    0.6,
    0.0,
    0.05,
    0.1,
    0.0,
    0.1,
    0.0,
    0.0,
]  # output of the neural network
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  # answer label

print(cross_entropy_error(np.array(y), np.array(t)))  # 0.510825457099338
```

모든 훈련데이터를 대상으로 손실 함수 값을 구하기(**미니배치**(mini-batch)):

- 일반적으로, 모든 데이터를 대상으로 손실 함수의 합을 구하는 것은 많은 시간이 소요된다. 고로 데이터 일부를 추려 전체의 ‘근사치’로 이용한다. 신경망 학습에서도 훈련 데이터로부터 일부만 골라 학습을 수행. 이 일부를 **미니배치**라고 한다.
- 교차 엔트로피 오차를 활용한 배치 → ‘평균 손실 함수’를 구하는 것

$$
E=-\frac{1}{N}\sum{n}\sum_{k}t_{nk}\log_e{y_{nk}}
$$

- $N$: 총 데이터 수
- $t_{nk}$: $n$번째 데이터의 $k$번째 값

<aside>

신경망을 학습할 떄 정확도를 지표로 삼아서는 안 된다. 정확도를 지표로 하면 매개변수의 미분이 대부분의 장소에서 0이 되기 때문이다.

즉, 정확도를 지표로 한다면, 매개변수 조정을 통해 정확도가 개선이 되지 않는 상황이나, 개선될지라도 정확도가 연속적인 변화를 띄지 않는 경우가 존재한다. 

</aside>

## 수치 미분(numerical differentiation)

- 미분은 ‘특정 순간’의 변화량을 뜻한다.
- 수치 미분은 아죽 작은 차분으로 미분하는 것.
- 수식을 전개해 미분하는 것은 ‘해석적(analytic)’이라는 말을 사용.

$$
\frac{df(x)}{dx} = \lim\nolimits_{h \to 0}{\frac{f(x+h) - f(x)}{h}}
$$

수치 미분은 컴퓨터에서 구현 시 아래와 같은 이슈가 있을 수 있다.

- **반올림 오차**(rounding error): $1e-50$을 float32형으로 나타내면 0.0이 되어, 올바로 표현 불가. $1e-4$정도에서 좋은 결과를 얻을 수 있음.
- 근사로 구한 접선이기 때문에 엄밀한 일치가 어려움.

오차를 줄이는 방법:

- $(x+h)$와 $(x-h)$일 때의 함수$f$ 의 차분을 계산하는 방법 → **중심 차분** 혹은 **중앙 차분**

구현:

```python
def numerical_diff(f, x):
	h = 1e-4 # 0.0001
	return (f(x+h) - f(x-h)) / (2*h)
```

수치 미분 예시:

$$
y = 0.01x^2+0.1x
$$

수치미분 구현:

```python
import numpy as np
import matplotlib.pyplot as plt

def function_1(x):
    return 0.01 * x**2 + 0.1 * x

def numerical_diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)

def tangent_line(f, x):
    d = numerical_diff(f, x)
    y = f(x) - d * x
    return lambda t: d * t + y

x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)

plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x, y)

plt.scatter(5, function_1(5))

tf = tangent_line(function_1, 5)
y2 = tf(x)
plt.plot(x, y2)

plt.show()
```

## 편미분(partial differentiation)

예시:

$$
f(x_0, x_1) = x_0^2+x_1^2
$$

**기울기**(gradient, 편미분을 벡터로 정리한 것) 구하기: 

$$
(\frac{\partial f}{\partial x_0}, \frac{\partial f}{\partial x_1})
$$

구현(이해하는데 난이도 있음):

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)  # generate array with same shape as x

    for idx in range(x.size):
        tmp_val = x[idx]
        # f(x+h)
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # f(x-h)
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val  # restore value

    return grad

# input is 1D array with 2 elements
def function_2(x):
    # return x[0] ** 2 + x[1] ** 2
    return np.sum(x**2)

x = np.linspace(-3, 3, 25)
y = np.linspace(-3, 3, 25)
X, Y = np.meshgrid(x, y)

Z = np.array(
    [
        [function_2(np.array([xi, yi])) for xi, yi in zip(row_x, row_y)]
        for row_x, row_y in zip(X, Y)
    ]
)

grad = np.array(
    [
        [
            numerical_gradient(function_2, np.array([xi, yi]))
            for xi, yi in zip(row_x, row_y)
        ]
        for row_x, row_y in zip(X, Y)
    ]
)

print(grad.shape)

# Extract gradient components
U = grad[:, :, 0]  # Gradient in x direction
V = grad[:, :, 1]  # Gradient in y direction

print(U.shape, V.shape)
print(U)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(X, Y, Z, cmap="viridis")

plt.show()

fig = plt.figure()
# Quiver graph
plt.quiver(X, Y, U, V)

plt.show()
```

## 경사법(경사 하강법)

매개변수 공간이 광대하여 어디가 최솟값이 되는 곳인지를 짐작하기 어려움. 이런 상황에서 기울기를 이용해 함수의 최솟값을 찾으려하는 것이 **경사법**(gradient method).

- 함수의 극솟값, 최솟값, 또 **안장점**(addle point)이 되는 장소에서는 기울기가 0
- 안장점은 어느 방향에서 보면 극댓값이고 다른 방향에서 보면 극솟값이 되는 점
- 경사법은 기울기가 0인 장소를 찾지만 그것이 반드시 최솟값이라고 할 수 없음
- 복잡하고 찌그러진 모양의 함수라면 평평한 곳으로 파고들면서 **고원**(plateau, 플래토)이라 하는, 학습이 진행되지 않는 정체기에 빠질 수 있음

경사법의 수식 표현:

$$
x_0 = x_0 - \eta\frac{\partial f}{\partial x_0}
$$

$$
x_1 = x_1 - \eta\frac{\partial f}{\partial x_1}
$$

- $\eta$ 기호(eta, 에타)는 갱신하는 양을 나타 냄.
- 이를 신경망 학습에서는 **학습률**(learning rate)이라고 함. → 즉 매개변수 값을 얼마나 갱신하느냐를 정하는 것
- 학습률 값은 특정 값으로 정해두어야 함. 일반적으로 너무 크거나 작으면 ‘좋은 장소’를 찾을 수 없음.
- 학습률 값을 변경하면서 올바르게 학습하고 있는지를 확인

구현:

```python
import numpy as np
import matplotlib.pyplot as plt

def function_2(x):
    # return x[0] ** 2 + x[1] ** 2
    return np.sum(x**2)

# Consider add type hints for numpy arrays
# reference: https://stackoverflow.com/questions/71109838/numpy-typing-with-specific-shape-and-datatype
def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]

        x[idx] = tmp_val + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val

    return grad

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    """
    init_x: initial value. 1D array.
    """
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x

def gradient_descent_recording(f, init_x, lr=0.01, step_num=100):
    """
    init_x: initial value. 1D array.
    """
    record = []
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
        record.append(x.copy())

    return np.array(record)

init_x = np.array([-3.0, 4.0])
record = gradient_descent_recording(function_2, init_x, lr=0.1, step_num=100)

fig = plt.figure()
plt.scatter(record[:, 0], record[:, 1])
plt.xlim(-3, 3)
plt.ylim(-4, 4)
plt.xlabel("X0")
plt.ylabel("X1")
plt.show()
```

학습률 같은 매개변수를 하이퍼파라미터(hyper parameter)라고 함.

- 직접 설정. 여러 후보 값 중에서 시험을 통해 가장 잘 학습하는 값을 찾아야 함

가중치, 편향 같은 신경망의 매개변수와는 성질이 다름.

- 가중치, 편향 → 훈련 데이터와 학습 알고리즘에 의해서 ‘자동’ 획득

## 학습 알고리즘 구현

<aside>

확률적 경사 하강법(stochastic gradient descent, SGD) → 미니배치를 무작위로 선정하기 때문

전제:

- 신경망에 적응 가능한 가중치와 편향이 있음.

1 단계 - 미니배치:

- 훈련 데이터 중 일부를 무작위로 가져 옴.

2 단계 - 기울기 산출:

- 미니배치의 손실 함수 값을 줄이기 위해 각 가중치 매개변수의 기울기를 구함

3 단계 - 매개변수 갱신:

- 가중치 매개변수를 기울기 방향으로 아주 조금 갱신

4 단계 - 반복:

- 1~3단계를 반복
</aside>

구현:

```python
from matplotlib import pyplot as plt
import numpy as np
import keras
from keras import utils

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))

def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    # TODO: Need to test it with a simple function
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]

        x[idx] = float(tmp_val) + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val  # restore value
        it.iternext()

    return grad

def get_data():
    Mnist = keras.datasets.mnist
    (x_train, t_train), (x_test, t_test) = Mnist.load_data()
    # Normalize and one-hot the image data
    return (
        x_train.reshape([-1, 28 * 28]) / 255,
        utils.to_categorical(t_train),
        x_test.reshape([-1, 28 * 28]) / 255,
        utils.to_categorical(t_test),
    )

class two_layer_net:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params["W1"] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params["b1"] = np.zeros(hidden_size)
        self.params["W2"] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params["b2"] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params["W1"], self.params["W2"]
        b1, b2 = self.params["b1"], self.params["b2"]

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y

    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads["W1"] = numerical_gradient(loss_W, self.params["W1"])
        grads["b1"] = numerical_gradient(loss_W, self.params["b1"])
        grads["W2"] = numerical_gradient(loss_W, self.params["W2"])
        grads["b2"] = numerical_gradient(loss_W, self.params["b2"])

        return grads

train_loss_list = []
train_acc_list = []
test_acc_list = []

iters_num = 10000
batch_size = 100
learning_rate = 0.1

x_train, t_train, x_test, t_test = get_data()
train_size = x_train.shape[0]

iter_per_epoch = max(train_size / batch_size, 1)

net = two_layer_net(input_size=784, hidden_size=50, output_size=10)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grad = net.numerical_gradient(x_batch, t_batch)

    for key in ("W1", "b1", "W2", "b2"):
        net.params[key] -= learning_rate * grad[key]

    loss = net.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    print(f"iteration {i}: loss {loss}")

    if i % iter_per_epoch == 0:
        train_acc = net.accuracy(x_train, t_train)
        test_acc = net.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(f"train acc, test acc | {train_acc}, {test_acc}")

plt.plot(np.arange(iters_num), train_loss_list)
plt.show()

```
