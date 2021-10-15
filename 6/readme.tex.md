# Kernel
## Introduction
기존의 regression/classification을 위한 linear parametic model은
* learning phase에서 (x, y)으로 구성된 dataset은 parameter vector를 예측하거나, [posterior distribution](https://en.wikipedia.org/wiki/Posterior_probability)를 결정하는데 사용된다.
    * posterior distribution가 뭔지 몰라 찾아봄. Intuition만 뽑아내자면, 데이터 분포로부터 확률분포를 예측하는 것. 흔히 사용되는 MLE도 그 일종인듯 하다.
    > From the vantage point of Bayesian inference, MLE is a special case of maximum a posteriori estimation (MAP) that assumes a uniform prior distribution of the parameters. (Wikipedia)
* 이후 새로운 input x로부터 y를 예측할 때, 기존의 dataset은 버려진다(discard).
* 하지만, **prediction phase에서도 기존의 training data points를 사용하는** pattern recognition technique이 있다.
    * Parzen probability density model (이게 뭐지)
    * neareset neighbours
    * 암튼 얘네 둘다 memory based method의 일종이란다.
        * 이 단어는 어떤 맥락에서 언급된 것일까? 그 반대는?
* 이들은 공통적으로 vector간 similarity를 구할 수 있는 metric이 요구된다.
* 많은 linear parametric model들은 "dual representation"으로 표현될 수 있는데, 이 과정에서 kernel function이 필요하다.

$$
k(x, x') = \phi(x)^T\phi(x')
$$
k는 symmetric funciton, 즉 $k(x, x') = k(x', x)
$이다.
* kernel은 feature space $\phi(x)$에 대해서 inner product로 표현된다.
    * kernel substitution이라 알려져 있는 일종의 trick을 적용할 수 있다.
    * input vector x가 scalar 곱으로 들어가면, 그 scalar 곱을 kernel로 치환할 수 있다.
### kernel trick
[cs229 강의](https://youtu.be/8NYoQiRANpg?t=1762)
1. write algorithm in terms of $<x^{(i)}, x^{(j)}>$
2. let that some mapping from $x\rightarrow\phi(x)$
    * $\phi(x)$의 차원이 너무 크면 계산 비용이 너무 많이 들겠죠?
3. find way to compute $k(x, z) = \phi(x)^T\phi(z)$
    * $\phi(x)$와 $\phi(z)$의 차원이 졸라게 크더라도 그것의 내적을 계산할 수 있는 트릭을 구하는 과정
4. Replace $<x, z>$ with $k(x, z)$

## 6.1. Dual Representation
다시 PRML로 돌아와서, SVM를 dual representation으로 나타내다 보면 이 과정에서 kernel 함수가 자연스럽게 보이게 되는데, 이를 알아보자.

### Ex. regularized sum-of-squares error
* why does formulation mean in statistical view? [link](./regularized_least_squares.md)


### Reference
1. MLE - https://en.wikipedia.org/wiki/Maximum_likelihood_estimation
