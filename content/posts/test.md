---
title: "テスト1"
date: 2021-01-06T12:23:01+09:00
draft: false
tags: ["test", "markdown"]
author: "akira"
---

テスト
tttsssttt

## Headings
blah

# H1
blah

###### H6
blah

## Paragraph
blah

## Tables
   Name | Age
--------|------
    Bob | 27
  Alice | 23

#### Inline Markdown within tables

| Italics   | Bold     | Code   |
| --------  | -------- | ------ |
| *italics* | **bold** | `code` |

## Code Blocks
#### Code block with backticks

```cpp
int main() {
  int y = SOME_MACRO_REFERENCE;
  int x = 5 + 6;
  cout << "Hello World! " << x << std::endl();
}
```

```python
def print_shape(name, dist, sample_shape=()):
    print(name, ":", "event shape:", dist.event_shape, "batch shape:", dist.batch_shape)
    print(name, ":", "sample shape", dist.sample(key, sample_shape=sample_shape).shape)
    print(name, ":", "whole shape:", dist.shape(sample_shape=sample_shape))
    print("")
```

## link
[link](https://www.google.com/)

## Math

Inline math: $ \quad p(\textbf{y}_n | \textbf{x}_n, \textbf{W}, \mu) = N(\textbf{y}_n | \textbf{W}^T \textbf{x}_n + \mu, \sigma^2_y \textbf{I}_D) $

Block math:
$$
 \varphi = 1+\frac{1} {1+\frac{1} {1+\frac{1} {1+\cdots} } } 
$$


## img
![test img](/posts/test/res.png)