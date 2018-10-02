---
layout: post
title: "「知乎」为什么 CSS 这么难学？"
subtitle: "Why I dislike CSS as a programming language"
author: "Hux"
header-img: "img/post-bg-css.jpg"
header-img-credit: "@WebdesignerDepot"
header-img-credit-href: "medium.com/@WebdesignerDepot/poll-should-css-become-more-like-a-programming-language-c74eb26a4270"
header-mask: 0.4
tags:
  - Web
  - CSS
---



### Lambda Change 구현

###### $relevance(term w | topic t) = λ * p(w | t) + (1 - λ) * p(w | t)/p(w)$


- Goal : topic 에 대해서 relevent 한 term을 계산해야한다.
- saliency(term w) = frequency(w) * [sum_t p(t | w) * log(p(t | w)/p(t))] for topics t
- relevance(term w | topic t) = $λ * p(w | t) + (1 - λ) * p(w | t)/p(w)$

##### Relevance 란?

Topic 안에 있는 Terms 을 평하는 기준이 됩니다. <br>
$p(w|t)$ 는 특정한 용어 w가 topic 에 등장할 확률입니다. <br>
$p(w)$ 은 특정 코퍼스의 marginal probability 입니다. <br>


##### Relevance (원문)

###### $relevance(term w | topic t) = λ * p(w | t) + (1 - λ) * p(w | t)/p(w)$


Definition
 - Let $\phi_{kw}$ denote the probability of term $w$ in $[1,..,V]$ for topic $k$ in $[1,..,K]$, where $V$ denote the number of terms in the vocabulary(전체 코퍼스).
 - Let $p_{w}$ denote the marginal probability of term $w$ in corpus.

Expaination
 - $\phi$ in LDA using Variational Bayes methods or Collapsed Gibbs sampling
 - $p_{w}$ emprical distribution of the corpus

Lambda
 - λ determines the weight given to the probability of term $w$ under topic $k$ relative to its lift.