# Implementation Spec: One-Sided Shampoo via Augmented Newton–Schulz

**Goal:** implement one-sided Shampoo so that the per-step update direction
$\mathbf{L}_t^{-1/2}\mathbf{M}_t$ is computed through the repo's `AugmentedNS` subroutine,
avoiding any SVD / `eigh` / coupled-Newton inverse-square-root in the hot path.

---

## 1. Background

One-sided Shampoo maintains a momentum buffer and a left preconditioner:

$$
\mathbf{M}_t = \beta_{t,1}\mathbf{M}_{t-1} + (1-\beta_{t,1})\mathbf{G}_t,
\qquad
\mathbf{L}_t = \beta_{t,2}\mathbf{L}_{t-1} + (1-\beta_{t,2})\mathbf{G}_t\mathbf{G}_t^\top,
$$

and updates the weights with

$$
\mathbf{W}_{t+1} = \mathbf{W}_t - \eta_t\,\mathbf{L}_t^{-1/2}\mathbf{M}_t .
$$

Shapes (for a weight matrix $\mathbf{W}\in\mathbb{R}^{m\times n}$):

| symbol | shape | meaning |
|---|---|---|
| $\mathbf{G}_t,\ \mathbf{M}_t,\ \mathbf{N}_t$ | $m\times n$ | gradient, momentum, residual factor |
| $\mathbf{L}_t,\ \mathbf{V}_t$ | $m\times m$ | left preconditioner / its residual |
| $\lambda_t$ | scalar | recursion scalar |

Forming $\mathbf{L}_t^{-1/2}$ by SVD is GPU-unfriendly, and coupled-Newton inverse-square-root
needs fp32/fp64 to stay stable. We instead reduce the update to a single call of a
matmul-only, low-precision-friendly Newton–Schulz subroutine.

> **One-sided convention.** The preconditioner acts on the *left* ($m$-dimension). To keep the
> $m\times m$ objects small, **precondition the smaller side**: if $m > n$, transpose
> $\mathbf{W}$ and $\mathbf{G}$ so the preconditioned dimension is the smaller one, run the
> optimizer, and transpose back.

---

## 2. The mathematical reduction to implement

### 2.1 The exact decomposition

There always exist a PSD matrix $\mathbf{V}_t \succeq 0$ and a scalar $\lambda_t > 0$ with

$$
\boxed{\ \mathbf{L}_t = \tfrac{1}{\lambda_t}\,\mathbf{M}_t\mathbf{M}_t^\top + \mathbf{V}_t\ }
$$

Substituting into the update gives the identity the implementation is built on:

$$
\mathbf{L}_t^{-1/2}\mathbf{M}_t
= \sqrt{\lambda_t}\,\bigl(\mathbf{M}_t\mathbf{M}_t^\top + \lambda_t\mathbf{V}_t\bigr)^{-1/2}\mathbf{M}_t
= \sqrt{\lambda_t}\;\mathsf{AugmentedNS}\!\bigl(\mathbf{M}_t,\ \lambda_t\mathbf{V}_t\bigr).
$$

(Check: $\mathbf{L}_t = \tfrac{1}{\lambda_t}(\mathbf{M}_t\mathbf{M}_t^\top + \lambda_t\mathbf{V}_t)$,
so $\mathbf{L}_t^{-1/2} = \sqrt{\lambda_t}\,(\mathbf{M}_t\mathbf{M}_t^\top + \lambda_t\mathbf{V}_t)^{-1/2}$.)

### 2.2 The recursion (maintain $\lambda_t,\mathbf{V}_t$ instead of $\mathbf{L}_t$)

Initialize $\mathbf{M}_0=0$, $\mathbf{V}_0=0$, and $\lambda_0=0$ (see §2.3).
For each step $t\ge 1$, with $\beta_1\equiv\beta_{t,1}$, $\beta_2\equiv\beta_{t,2}$:

$$
\lambda_t = \frac{\beta_{1}^2}{\beta_{2}}\,\lambda_{t-1} + \frac{(1-\beta_{1})^2}{1-\beta_{2}},
$$

$$
\mathbf{N}_t = \sqrt{\frac{\beta_{1}^2\,\lambda_{t-1}}{\beta_{2}\,\lambda_t}}\;\mathbf{G}_t
\;-\;\frac{1-\beta_{1}}{1-\beta_{2}}\sqrt{\frac{\beta_{2}}{\lambda_{t-1}\,\lambda_t}}\;\mathbf{M}_{t-1},
$$

$$
\mathbf{V}_t = \beta_{2}\,\mathbf{V}_{t-1} + (1-\beta_{2})\,\mathbf{N}_t\mathbf{N}_t^\top .
$$

Then $\mathbf{L}_t = \tfrac{1}{\lambda_t}\mathbf{M}_t\mathbf{M}_t^\top + \mathbf{V}_t$ holds exactly for all $t\ge 1$.
$\mathbf{N}_t$ is $m\times n$ (same shape as $\mathbf{G}_t$), so $\mathbf{N}_t\mathbf{N}_t^\top$ is $m\times m$.

### 2.3 Choosing $\lambda_0$ (and why $\lambda_0 = 0$ is the convenient default)

- **Invariance.** Any $\lambda_0 \ge 0$ produces the **same $\mathbf{L}_t$ and the same update**;
  it only reparametrizes how each $\mathbf{L}_t$ is split between $\tfrac{1}{\lambda_t}\mathbf{M}_t\mathbf{M}_t^\top$
  and $\mathbf{V}_t$. $\lambda_0$ is therefore *not* a hyperparameter to tune.
- **Use $\lambda_0 = 0$.** Then $\lambda_1 = (1-\beta_1)^2/(1-\beta_2) > 0$ and $\mathbf{V}_1 = 0$, so the
  **first update is pure orthogonalization**, $\mathbf{L}_1^{-1/2}\mathbf{M}_1 \propto (\mathbf{M}_1\mathbf{M}_1^\top)^{-1/2}\mathbf{M}_1$
  (the Muon direction); the residual $\mathbf{V}_t$ then accumulates from step 2 onward.
- **One numerical guard at $t=1$.** When $\lambda_{t-1}=0$, the $\mathbf{M}_{t-1}$ coefficient
  $\sqrt{\beta_2/(\lambda_{t-1}\lambda_t)}$ is $+\infty$, but it multiplies $\mathbf{M}_0=0$. Compute it as
  $0$ (i.e. drop that term) so you never evaluate $\infty\cdot 0$. The $\mathbf{G}_t$ coefficient is
  $\sqrt{\beta_1^2\lambda_0/(\beta_2\lambda_1)}=0$ as well, so $\mathbf{N}_1=0$. This guard is only
  ever needed at $t=1$, since $\lambda_t>0$ for all $t\ge 1$.

### 2.4 Other properties to assert

- **Positivity.** For $\lambda_0\ge 0$ all square-root arguments stay $\ge 0$, and $\lambda_t>0$ for $t\ge1$.
- **Boundedness.** With constant betas, $\lambda_t$ converges iff $\beta_1^2 < \beta_2$
  (e.g. $\beta_1=0.9,\beta_2=0.95\Rightarrow 0.81<0.95$ ✓). If $\beta_1^2\ge\beta_2$, $\lambda_t$ grows;
  still valid mathematically, but warn the user and consider clamping.
- **Memory.** You store $\mathbf{V}_t$ ($m\times m$), $\mathbf{M}_t$ ($m\times n$), $\lambda_t$ (scalar) —
  the **same** footprint as storing $\mathbf{L}_t$. The win is *compute*, not memory.

---

## 3. The `AugmentedNS` subroutine (black box)

Treat this as a high-level routine provided by the repo. The optimizer only needs its contract:

```
AugmentedNS(M, A) -> P
  in:  M (m x n),  A (m x m, symmetric PSD)
  out: P (m x n)  =  (M Mᵀ + A)^(-1/2) M
```

All internal concerns — the Newton–Schulz iteration itself, iteration count, scaling/normalization,
working precision, and handling of near-singular / rank-deficient $\mathbf{A}$ (e.g. the early-step
$\mathbf{A}=\lambda_1\mathbf{V}_1=0$ case) — live **inside** the subroutine and are out of scope here.

---

## 4. The full optimizer step

For each parameter tensor (reshape >2D tensors to 2D; transpose so $m\le n$):

```
# state:  M (m x n) = 0,  V (m x m) = 0,  lambda = 0.0
# on step t with gradient G, lr eta, betas (b1, b2):

lam_prev = lambda
lambda   = (b1*b1 / b2) * lam_prev + (1 - b1)**2 / (1 - b2)

# N_t  — uses M and lambda from BEFORE their update (i.e. M_{t-1}, lambda_{t-1})
cg = sqrt( (b1*b1 * lam_prev) / (b2 * lambda) )
if lam_prev == 0:
    N = cg * G                     # cg == 0 here; M-term dropped since M_{t-1}=0  ->  N = 0
else:
    cm = ((1 - b1) / (1 - b2)) * sqrt( b2 / (lam_prev * lambda) )
    N  = cg * G - cm * M

# update residual V and momentum M
V = b2 * V + (1 - b2) * (N @ N.T)
M = b1 * M + (1 - b1) * G

# update direction:  L^{-1/2} M  =  sqrt(lambda) * AugmentedNS(M, lambda * V)
dir = sqrt(lambda) * AugmentedNS(M, lambda * V)

W -= eta * dir
```

**Ordering matters:** compute $\lambda_t$ and $\mathbf{N}_t$ from $\mathbf{M}_{t-1}$ and
$\lambda_{t-1}$ **before** overwriting $\mathbf{M}$ and $\lambda$. Square-root coefficients
should be computed in fp32.

---

## 5. Suggested file/API layout

- `one_sided_shampoo.py` — a `torch.optim.Optimizer` subclass holding per-param state
  `{M, V, lambda}`; handles reshaping, the $m\le n$ transpose, dtype, and the step in §4.
  Imports `AugmentedNS` from the repo.
- Config: `beta1, beta2, lr, lambda0 (= 0.0)`.
- Reference (test-only) `one_sided_shampoo_reference.py`: builds $\mathbf{L}_t$ directly by its
  EMA and computes $\mathbf{L}_t^{-1/2}$ via `torch.linalg.eigh` (clamp eigenvalues $\ge 0$).

---

## 6. Tests the implementation must pass

1. **Decomposition identity.** Drive random gradients for $t=1..T$. At each step assert
   $\big\|\mathbf{L}_t - (\tfrac{1}{\lambda_t}\mathbf{M}_t\mathbf{M}_t^\top + \mathbf{V}_t)\big\| / \|\mathbf{L}_t\|$
   is at machine-precision level, where $\mathbf{L}_t$ is built by its own EMA and
   $(\lambda_t,\mathbf{V}_t)$ by the recursion (with $\lambda_0=0$).
2. **$\lambda_0$-invariance.** Run with $\lambda_0\in\{0, 0.1, 1, 10\}$; the reconstructed
   $\mathbf{L}_t$ and the final update direction must match across all choices.
3. **First-step sanity.** With $\lambda_0=0$, assert $\mathbf{V}_1=0$ and that the step-1 direction
   equals (up to the $\sqrt{\lambda_1}$ scaling) $\mathsf{AugmentedNS}(\mathbf{M}_1,\mathbf{0})$.
4. **End-to-end step.** One optimizer step's update direction must match $\mathbf{L}_t^{-1/2}\mathbf{M}_t$
   from the `eigh` reference to the subroutine's tolerance.
5. **Convergence smoke test.** Minimize a small quadratic / fit a tiny MLP; loss should decrease
   comparably to the `eigh`-based reference.

---

## 7. Pitfalls checklist

- [ ] Preconditioned dimension is the **smaller** one (transpose when $m>n$).
- [ ] $\mathbf{N}_t$ and $\lambda_t$ use $\mathbf{M}_{t-1}$ and $\lambda_{t-1}$ (pre-update values).
- [ ] $\lambda_0=0$ default; guard the $\mathbf{M}_{t-1}$ term when $\lambda_{t-1}=0$ to avoid $\infty\cdot 0$ (→ NaN).
- [ ] Square-root coefficients computed in fp32.
- [ ] `AugmentedNS` is called as `AugmentedNS(M, lambda*V)`, and the result is scaled by $\sqrt{\lambda_t}$.
- [ ] Warn if $\beta_1^2 \ge \beta_2$ (unbounded $\lambda_t$).
