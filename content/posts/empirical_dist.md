---
title: "経験分布とは？"
date: 2022-01-07T12:59:05+09:00
draft: false
tags: ["機械学習", "Machine Learning"]
author: "akira"
---

# "The Dirac Distribution and Empirical Distribution"


↑ は、[Goodfellow本 3.9.5節](https://www.deeplearningbook.org/contents/prob.html)のタイトルです。Empirical Distribution が「経験分布」です。

経験分布はデータのサンプル（=観測・経験）集合の確率分布です。最尤推定はつまるところ、この経験分布と想定されるモデルの分布（正規分布、ベルヌーイ、カテゴリカルなど）の間のKLダイバージェンスを最小化するように訓練することです。そしてKLダイバージェンスを最小化するということは、交差エントロピーを最小化することと同じです。（この段でMSEは経験分布とガウス分布をモデルに仮定した場合の交差エントロピーだということがわかる）（ [Goodfellow本 5.5節](https://www.deeplearningbook.org/contents/ml.html) あたりに書いてある)

何が言いたいかというと、経験分布とモデル分布の比較として最尤推定を捉えれば、回帰・2クラス分類・多クラス分類の”各種”誤差関数の根っこは同じになって、全部一貫して理解できるので、嬉しい、ということです。

[ESLの2.4](https://www.amazon.co.jp/%E7%B5%B1%E8%A8%88%E7%9A%84%E5%AD%A6%E7%BF%92%E3%81%AE%E5%9F%BA%E7%A4%8E-%E2%80%95%E3%83%87%E3%83%BC%E3%82%BF%E3%83%9E%E3%82%A4%E3%83%8B%E3%83%B3%E3%82%B0%E3%83%BB%E6%8E%A8%E8%AB%96%E3%83%BB%E4%BA%88%E6%B8%AC%E2%80%95-Trevor-Hastie/dp/432012362X/ref=sr_1_1?adgrpid=73121729924&gclid=Cj0KCQiAoY-PBhCNARIsABcz773CoHk8_rfSNDRupEaGt6GY-zhqME-uzM32RsvJ5JK5J2CeFppdi58aAudYEALw_wcB&hvadid=353826168644&hvdev=c&hvlocphy=1009301&hvnetw=g&hvqmt=e&hvrand=5666640908829365875&hvtargid=kwd-848857017784&hydadcr=13360_11184391&jp-ad-ap=0&keywords=%E7%B5%B1%E8%A8%88%E7%9A%84%E5%AD%A6%E7%BF%92%E3%81%AE%E5%9F%BA%E7%A4%8E&qid=1642387895&sr=8-1)では統計的決定理論として、経験分布と予測分布（=>モデル分布のこと）の誤差として、最小二乗法と最近傍法を一般化する形で説明されてます。

このあたりの首尾一貫した統計的学習理論は、僕みたいにエンジニアとして予備知識なしにハンズオン系の機械学習本から入った場合、少なくとも初回は全く素通りすることになってしまい、例えば最小二乗法と最近傍法は全く別の「ツール」として頭に入ってきて、先に進むにつれてそんなツールがひたすら増えて、もうダメです状態になります。何度か諦めながら色々と手を出しているうちに線がつながってくるのですが、結構時間も無駄にします。ただ、誰かが組んだカリキュラムに沿って最後まで登り切る忍耐もないし、結局いろんなところをうろうろしたうえで、来るべき時期が来ないとつながるべき線もつながらないという。まぁそういうもんですかね。。。

話がそれました。そんな経験分布ですが、

1. 経験分布は、
2. ディラック分布を構成要素としていて、
3. それはディラックのデルタ関数により確率密度関数として定義される

なのですが、読んでるだけだとそもそも３のディラックのデルタ関数のピンが来ない。理解のためにこんな感じかという実装をしてみました。

サンプルが正規分布してるとして、

```python
import jax.numpy as jnp
from jax import random
import numpyro.distributions as dist
import matplotlib.pyplot as plt; 
key = random.PRNGKey(49)
```
```python
normal = dist.Normal(loc=0.5, scale=1.)
samples = normal.sample(key, (1000,))
plt.hist(samples)
plt.show()
```

![dist](/posts/empirical_dist/output.png)

このサンプルを確率分布にしたものが、経験分布です。なので、

```python
def dirac_delta_fn(xx, sigma):
   val = []
   for x in xx:
       if -(1 / (2 * sigma)) <= x and x <= (1 / (2 * sigma)):
           val.append(sigma)
       else:
           val.append(0)
   return jnp.array(val)

def pick_prob_of_x(current_x, samples, sigma=0.5):
    return jnp.sum(dirac_delta_fn([current_x - sample for sample in samples], sigma)) / samples.shape[0]    

def form_empirical_distribution(_from, _to, samples, step=0.01):
    xx = []
    yy = []
    for current_x in np.arange(_from, _to, step):
        y = pick_prob_of_x(current_x, samples)
        xx.append(current_x)
        yy.append(y)
    return xx, yy

xx, yy = form_empirical_distribution(-5, 5, samples)

plt.plot(xx, yy)
plt.show()
```

![dist](/posts/empirical_dist/output2.png)

サンプルだけから確率分布を作ることができるのであった。