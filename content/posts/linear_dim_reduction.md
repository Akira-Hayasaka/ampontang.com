---
title: "線形因子モデル"
date: 2022-01-09T14:53:39+09:00
draft: false
tags: ["統計的推測", "numpyro", "機械学習", "潜在変数", "latent space"]
author: "akira"
---

## 線形因子モデル

線形因子モデルは、[goodfellow本](https://www.deeplearningbook.org/)ではその後に続く深層生成モデルの、[須山本](https://www.amazon.co.jp/%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E3%82%B9%E3%82%BF%E3%83%BC%E3%83%88%E3%82%A2%E3%83%83%E3%83%97%E3%82%B7%E3%83%AA%E3%83%BC%E3%82%BA-%E3%83%99%E3%82%A4%E3%82%BA%E6%8E%A8%E8%AB%96%E3%81%AB%E3%82%88%E3%82%8B%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E5%85%A5%E9%96%80-KS%E6%83%85%E5%A0%B1%E7%A7%91%E5%AD%A6%E5%B0%82%E9%96%80%E6%9B%B8-%E9%A0%88%E5%B1%B1-%E6%95%A6%E5%BF%97/dp/4061538322/ref=sr_1_1?keywords=%E3%83%99%E3%82%A4%E3%82%BA%E6%8E%A8%E8%AB%96%E3%81%AB%E3%82%88%E3%82%8B%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E5%85%A5%E9%96%80&qid=1641542845&sprefix=%E3%83%99%E3%82%A4%E3%82%BA%E6%8E%A8%E8%AB%96%2Caps%2C185&sr=8-1)では第五章の応用モデルの、基礎を成しているとのことなので、やってみました。モデルとしては、[PRML](https://www.amazon.co.jp/%E3%83%91%E3%82%BF%E3%83%BC%E3%83%B3%E8%AA%8D%E8%AD%98%E3%81%A8%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92-%E4%B8%8B-%E3%83%99%E3%82%A4%E3%82%BA%E7%90%86%E8%AB%96%E3%81%AB%E3%82%88%E3%82%8B%E7%B5%B1%E8%A8%88%E7%9A%84%E4%BA%88%E6%B8%AC-C-M-%E3%83%93%E3%82%B7%E3%83%A7%E3%83%83%E3%83%97/dp/4621061240/ref=pd_lpo_1?pd_rd_i=4621061240&psc=1)の12章を参考にしています。

線形因子モデルは、潜在変数に情報を詰め込んで、そこからデータを再現します。

- PPCA, factor analysisは大体これと同じ
- ICAはローカルの潜在変数に独立を仮定し、ガウス分布以外の物を使う

ので、goodfellow本13章で紹介されている発展的なモデルも、これができればそんなに遠くない（はず）。

## 潜在変数の解釈

潜在変数というのは、深層学習の文脈で表現（representation）や特徴（feature）と呼ばれるもので、
- ベイズ系の統計推測での潜在変数
- NNが隠れ層で学習する[多様体](http://colah.github.io/posts/2014-03-NN-Manifolds-Topology/)
- オートエンコーダの隠れ層
- スパースコーディングの成果物
- 教師なし学習で事前学習する（していた）際の目的
- CNNのフィルタが学習するもの
- RNNが時系列で共有するもの（パラメータシェアリング）

などなどは、同じイデアを共有していて、緩やかに繋がっているものだと理解しています。深層学習ではその潜在変数（=表現・特徴）がスパースであるほど統計的に意義がある（uniformに分布してると何も言えないから）かつその潜在変数を使った後続の回帰や分類タスクの精度が上がるので、そのスパース性を求めて様々な正則化が適用されます。

この潜在変数をいかにうまく設計するかが、統計的推論・統計的機械学習のキモです。

## 実装

### グラフィカルモデル
![model](/posts/linear_dim_reduction/model.jpg)

### 観測データ

$$
    \quad Y=[y_1,...,y_n] \quad y_n \in \mathbb{R}^D
$$

### 潜在変数

$$
    \quad X=[x_1,...,x_n] \quad x_n \in \mathbb{R}^M \\\\
    \quad \textbf{W} \in \mathbb{R}^{M \times D} \quad (\textbf{W}_d \in \mathbb{R}^M \quad W の d 番⽬の列ベクトル)\\\\
    \quad \mu \in \mathbb{R}^D
$$

### パラメータ

$$
    \quad \sigma^2_y \in \mathbb{R}^+ \\\\
    \Sigma_w \\\\
    \Sigma_{\mu}
$$

### 個別の分布

$$
    p(\textbf{W}) = \prod^D_{d=1} N(\textbf{W}_d | \textbf{0}, \Sigma_w) 
$$

$$
    p(\mu) = N(\mu | \textbf{0}, \Sigma_{\mu}) 
$$

$$
    p(\textbf{x}_n) = N(\textbf{x}_n | \textbf{0}, \textbf{I}_M)
$$

### $\textbf{y}_n$ の条件付き分布

$$
    p(\textbf{y}_n | \textbf{x}_n, \textbf{W}, \mu) = N(\textbf{y}_n | \textbf{W}^T \textbf{x}_n + \mu, \sigma^2_y \textbf{I}_D) \\
$$

### 同時分布
$$
    p(\textbf{Y}, \textbf{X}, \textbf{W}, \mu) = p(\textbf{W})p(\mu)\prod^N_{n=1}p(\textbf{y}_n | \textbf{x}_n, \textbf{W}, \mu)p(\textbf{x}_n) \\
$$

### model

```python
def print_shape(name, dist, sample_shape=()):
    print(name, ":", "event shape:", dist.event_shape, "batch shape:", dist.batch_shape)
    print(name, ":", "sample shape", dist.sample(key, sample_shape=sample_shape).shape)
    print(name, ":", "whole shape:", dist.shape(sample_shape=sample_shape))
    # print(name, ":", "sample", dist.sample(key, sample_shape=sample_shape))
    print("")

def model(D, M, N, obs=None, debug=False):
    # mu
    loc_mu = jnp.full(D, 0) #jax.random.normal(key, (D,))
    scale_mu = jnp.full(D, 1)
    dist_mu = dist.Normal(loc=loc_mu, scale=scale_mu).to_event()
    mu = numpyro.sample("latent_mu", dist_mu)
    if (debug):
        print_shape("dist_mu", dist_mu)

    # W
    loc_W = jnp.full((M, D), 0) #jax.random.normal(key, (M,D))
    scale_W = jnp.full((M, D), 1)
    dist_W = dist.Normal(loc=loc_W, scale=scale_W).to_event()
    W = numpyro.sample("latent_W", dist_W)
    if (debug):
        print_shape("dist_W", dist_W)

    # X, latent
    loc_x = jnp.full((N, M), 0) #jax.random.normal(key, (N,M))       
    scale_x = jnp.full((N, M), 1)        
    dist_X = dist.Normal(loc=loc_x, scale=scale_x)
    X = numpyro.sample("latent_x", dist_X)
    if (debug):
        print_shape("dist_X", dist_X)

    # Y
    loc_Y = jnp.zeros((N, D))
    for i in range(N):
        loc_Y = loc_Y.at[i].set(jnp.dot(W.T, X[i]) + mu)
    sacle_Y = jnp.full_like(loc_Y, 1)
    dist_Y = dist.Normal(loc=loc_Y, scale=sacle_Y)
    Y = numpyro.sample("Y_obs", dist_Y, obs=obs)
    if (debug):
        print("sacle_Y.shape", sacle_Y.shape)
        print("loc_Y.shape", loc_Y.shape)
        print_shape("dist_Y", dist_Y)

D, M, N = 64*64, 32, 10
print("D:", D, "M:", M, "N:", N, "\n")
prior_model_trace = handlers.trace(handlers.seed(model, key))
prior_model_exec = prior_model_trace.get_trace(D=D, M=M, N=N, obs=None, debug=True)
```

### olivetti face dataset

```python
import pandas as pd
from sklearn import datasets
from skimage.transform import rescale
from skimage import data, color

data = datasets.fetch_olivetti_faces()
df = pd.DataFrame(data.data)
print(df.shape)
df.head()

N = 9
img_res = 64

rndidx = np.random.choice(df.shape[0], N)
imgs = jnp.zeros((len(rndidx), df.shape[1]))
for i in range(len(rndidx)):
    imgs = imgs.at[i].set(df.loc[rndidx[i]].values)

col, row = int(round(np.sqrt(N))), int(round(np.sqrt(N)))
fig = plt.figure(figsize=(10, 10))
for i in range(1, col*row+1):
    fig.add_subplot(row, col, i)
    plt.gray() 
    plt.imshow(imgs[i-1].reshape(img_res, img_res))
    plt.grid(None)
plt.show()
```
![model](/posts/linear_dim_reduction/output0.png)

### reduce 4096 to 0.005% (20)

```python
D = imgs.shape[1]
M = int(round(imgs.shape[1] * 0.005))
print("reduce", D, "to", M)
```

```python
guide = numpyro.infer.autoguide.AutoDelta(model)

optimizer = numpyro.optim.Adam(step_size=0.0005)
svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
svi_result = svi.run(key, 2000, D=D, M=M, N=N, obs=jnp.array(imgs))
params = svi_result.params
pp.pprint(params)
```

### 復元

```python
expected_mu = params["latent_mu_auto_loc"]
expected_W = params["latent_W_auto_loc"]
expected_x_n = params["latent_x_auto_loc"]
print("expected_mu.shape", expected_mu.shape)
print("expected_W.shape", expected_W.shape)
print("expected_x_n.shape", expected_x_n.shape)

imgs_reconstructed = jnp.zeros((N, D))
for i in range(N):
    x_n = expected_x_n[i]
    imgs_reconstructed = imgs_reconstructed.at[i].set(jnp.dot(expected_W.T, x_n) + expected_mu)

col, row = int(round(np.sqrt(N))), int(round(np.sqrt(N)))
fig = plt.figure(figsize=(10, 10))
for i in range(1, col*row+1):
    fig.add_subplot(row, col, i)
    plt.imshow(imgs_reconstructed[i-1].reshape(img_res, img_res))
    plt.grid(None)
plt.gray()    
plt.show()
```

![model](/posts/linear_dim_reduction/output1.png)

