# A Tiny Paper on Continuous Functions

## Background

We work in the standard real analysis setting on the closed interval [0, 1].
A function $f : [0, 1] \to \mathbb{R}$ is *continuous* in the usual epsilon-delta
sense.

## Assumption 1

Throughout the paper we assume that $f$ is continuous on $[0, 1]$.

## Lemma 1.

If $f$ is continuous on $[0, 1]$, then $f$ is bounded on $[0, 1]$.

Proof. By Assumption 1 the function $f$ is continuous on the closed interval
$[0, 1]$. The Extreme Value Theorem gives a maximum $M$ and minimum $m$ such
that $m \le f(x) \le M$ for all $x \in [0, 1]$, hence $f$ is bounded.
$\blacksquare$

## Theorem 1.

If $f$ is continuous on $[0, 1]$, then $\int_0^1 f(x)\,dx$ exists.

Proof. By Lemma 1, $f$ is bounded on $[0, 1]$. A bounded continuous function on
a closed interval is Riemann integrable, so the integral $\int_0^1 f(x)\,dx$
exists. $\blacksquare$

## Experiment

We numerically integrate $f(x) = x^2$ via the trapezoidal rule with $n = 1000$
subintervals and obtain $0.3333\overline{3}$, matching the analytic value
$1/3$.
