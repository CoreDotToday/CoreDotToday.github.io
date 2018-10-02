---
layout: post
title: "「LDA와 토픽 모델링에 대한 기본적인 이해"
subtitle: "테라비이트의 텍스트 문서를 분석하는 방법"
author: "도승헌"
header-img: "img/post-bg-halting.jpg"
header-mask: 0.3
tags:
  - LDA
  - Gensim
  - 토픽모델링
---

> 본 문서는 다음과 같은 사이트를 참고하였습니다.
> [Gensim](https://radimrehurek.com/gensim/about.html)
> [ratsgo - 토픽모델링](https://ratsgo.github.io/from%20frequency%20to%20semantics/2017/06/01/LDA/)

데이터마이닝의 한 분야인 텍스트마이닝은 텍스트문서, 이메일, HTML문서와 같이 비구조화(Unstructured), 혹은 반구조화(Semi-Structured) 된 텍스트문서에서<br>
새로운 정보를 추출하는 정보기술로, 토픽모델링은 텍스트 마이닝에서 사용하는 연구방법론이다. 가장 대표적인 토픽모델링 기법인 Blei etal.(2003)의<br>
LDA(Latent Dirichlet Allocation)은 다수의 문서에서 잠재적으로 의미 있는 토픽을 발견하는 절차적 확률 분포 모델이다. LDA는 단어들의 집합이<br>
어떤 토픽들로 묶인다고 가정하고, 이 단어들이 각각의 토픽에 구성될 확률을 계산하여 결과 값을 토픽에 해당할 가능성이 높은 단어들의 집합으로 추출하는 방식이다.<br>

## Gensim이란?
주어진 기사에 가장 유사한 기사를 생성하는 기능을 구현하고 싶었던 개발자들이 만들어낸 라이브러리 입니다.(gensim = “generate similar”) <br> “Latent Semantic Methods”을 개발하고자 시작했던
我们假设存在这么一个「停机程序」，不管它是怎么实现的，但是它能够回答「停机问题」：它接受一个「程序」和一个「输入」，然后判断这个「程序」在这个「输入」下是否能给出结果：

```py
def is_halt(program, input) -> bool:
  # 返回 True  如果 program(input) 会返回
  # 返回 False 如果 program(input) 不返回
```

（在这里，我们通过把一个函数作为另一个函数的输入来描述一个「程序」作为另一个「程序」的「输入」，如果你不熟悉「头等函数」的概念，你可以把所有文中的函数对应为一个具备该函数的对象。）

为了帮助大家理解这个「停机程序」的功能，我们举个使用它的例子：

```py
from halt import is_halt

def loop():
  while(True):
      pass

# 如果输入是 0，返回，否则无限循环
def foo(input):
  if input == 0:
    return
  else:
    loop()

is_halt(foo, 0)  # 返回 True
is_halt(foo, 1)  # 返回 False
```

是不是很棒？

不过，如果这个「停机程序」真的存在，那么我就可以写出这么一个「hack 程序」：

```py
from halt import is_halt

def loop():
  while(True):
      pass

def hack(program):
  if is_halt(program, program):
    loop()
  else:
    return
```

这个程序说，如果你 `is_halt` 对 `program(program)` 判停了，我就无限循环；如果你判它不停，我就立刻返回。

那么，如果我们把「hack 程序」同时当做「程序」和「输入」喂给「停机程序」，会怎么样呢？

```py
is_halt(hack, hack)
```

你会发现，如果「停机程序」认为 `hack(hack)` 会给出结果，即 `is_halt(hack, hack)`) 返回 `True`) ，那么实际上 `hack(hack)` 会进入无限循环。而如果「停机程序」认为 `hack(hack)` 不会给出结果，即 `is_halt(hack, hack)` 返回 ，那么实际上 `hack(hack)` 会立刻返回结果……

这里就出现了矛盾和悖论，所以我们只能认为，我们最开始的假设是错的：

**这个「停机程序」是不存在的。**

## 意义

简单的说，「停机问题」说明了现代计算机并不是无所不能的。

上面的例子看上去是刻意使用「自我指涉」来进行反证的，但这只是为了证明方便。实际上，现实中与「停机问题」一样是现代计算机「不可解」的问题还有很多，比如所有「判断一个程序是否会在某输入下怎么样？」的算法、Hilbert 第十问题等等，wikipedia 甚至有一个 [List of undecidable problems](https://link.zhihu.com/?target=https%3A//en.wikipedia.org/wiki/List_of_undecidable_problems)。

## 漫谈

如果你觉得只是看懂了这个反证法没什么意思：

1.  最初图灵提出「停机问题」只是针对「图灵机」本身的，但是其意义可以被推广到所有「算法」、「程序」、「现代计算机」甚至是「量子计算机」。
2.  实际上「图灵机」只能接受（纸带上的）字符串，所以在图灵机编程中，无论是「输入」还是另一个「图灵机」，都是通过编码来表示的。
3.  「图灵机的计算能力和现代计算机是等价的」，更严谨一些，由于图灵机作为一个假象的计算模型，其储存空间是无限大的，而真实计算机则有硬件限制，所以我们只能说「不存在比图灵机计算能力更强的真实计算机」。
4.  这里的「计算能力」（power）指的是「能够计算怎样的问题」（capablity）而非「计算效率」（efficiency），比如我们说「上下文无关文法」比「正则表达式」的「计算能力」强因为它能解决更多的计算问题。
5.  「图灵机」作为一种计算模型形式化了「什么是算法」这个问题（[邱奇－图灵论题](https://link.zhihu.com/?target=https%3A//en.wikipedia.org/wiki/Church%25E2%2580%2593Turing_thesis)）。但图灵机并不是唯一的计算模型，其他计算模型包括「[Lambda 算子](https://link.zhihu.com/?target=https%3A//en.wikipedia.org/wiki/Lambda_calculus)」、[μ-递归函数](https://link.zhihu.com/?target=https%3A//en.wikipedia.org/wiki/%25CE%259C-recursive_function)」等，它们在计算能力上都是与「图灵机」等价的。因此，我们可以用「图灵机」来证明「[可计算函数](https://link.zhihu.com/?target=https%3A//en.wikipedia.org/wiki/Computable_function)」的上界。也因此可以证明哪些计算问题是超出上界的（即不可解的）。
6.  需要知道的是，只有「可计算的」才叫做「算法」。
7.  「停机问题」响应了「哥德尔的不完备性定理」。

## 拓展阅读：

中文：

- [Matrix67: 不可解问题(Undecidable Decision Problem)](https://link.zhihu.com/?target=http%3A//www.matrix67.com/blog/archives/55)

- [Matrix67: 停机问题、Chaitin 常数与万能证明方法](https://link.zhihu.com/?target=http%3A//www.matrix67.com/blog/archives/901)

- [刘未鹏：康托尔、哥德尔、图灵--永恒的金色对角线(rev#2) - CSDN 博客](https://link.zhihu.com/?target=http%3A//blog.csdn.net/pongba/article/details/1336028)

- [卢昌海：Hilbert 第十问题漫谈 (上)](https://link.zhihu.com/?target=http%3A//www.changhai.org/articles/science/mathematics/hilbert10/1.php)

英文：

- [《Introduction to the Theory of Computation》](https://link.zhihu.com/?target=https%3A//en.wikipedia.org/wiki/Introduction_to_the_Theory_of_Computation)

- [Turing Machines Explained - Computerphile](https://link.zhihu.com/?target=https%3A//www.youtube.com/watch%3Fv%3DdNRDvLACg5Q)

- [Turing & The Halting Problem - Computerphile](https://link.zhihu.com/?target=https%3A//www.youtube.com/watch%3Fv%3DmacM_MtS_w4%26t%3D29s)

- [Why, really, is the Halting Problem so important?](https://link.zhihu.com/?target=https%3A//cs.stackexchange.com/questions/32845/why-really-is-the-halting-problem-so-important)
