# Heston Neural Pricer & Calibration

**Oxford Mathematical and Computational Finance — Deep Learning | March 2025**

End-to-end deep learning pipeline for pricing and calibrating European options under the Heston stochastic volatility model. The project is structured in three sequential parts: Monte Carlo dataset generation, neural network surrogate training, and gradient-based calibration using the surrogate as a differentiable pricer.

---

## Project Structure

```
├── 1a_heston_monte_carlo_simulation.ipynb   # Part 1a — Heston MC simulator & dataset generation
├── 1b_heston_mlp_surrogate_pricer.ipynb     # Part 1b — MLP surrogate pricer training & evaluation
├── 2_heston_calibration_neural_network.ipynb # Part 2  — NN-based calibration from a price surface
├── consigne.pdf                              # Exam problem statement
└── report.pdf                               # Submitted exam report
```

---

## Motivation — Why a Neural Network Pricer?

Pricing a single option under the Heston model via Monte Carlo requires simulating thousands of stochastic paths, each with hundreds of time steps. A single price evaluation takes on the order of seconds on CPU. This is manageable for one-off pricing, but becomes a hard bottleneck in two practical situations:

**1. Calibration.** Fitting the Heston model to a market surface (say, 100+ observed option prices) requires solving a non-linear least-squares problem over 5 parameters. Each iteration of the optimizer calls the pricer once per contract. With a gradient-free solver, this means thousands of MC evaluations — each noisy and slow. The calibration can take minutes to hours, and MC noise corrupts the gradient signal.

**2. Real-time or large-scale pricing.** Risk systems, trading desks, and scenario engines need to reprice entire books — potentially millions of contracts per day — with consistent parameters. MC is simply too slow for this.

The neural network surrogate solves both problems at once. Once trained offline on a large synthetic dataset, the MLP evaluates in microseconds per contract and is fully differentiable. This unlocks:

- **Gradient-based calibration**: backpropagate directly through the pricer to compute $\partial \mathcal{L} / \partial \theta$ analytically, replacing noisy finite-difference gradients with exact ones. Calibration converges faster and more reliably.
- **Instant inference**: after calibration, reprice any contract in the parameter neighborhood at negligible cost.
- **Scalability**: batch evaluation on GPU handles millions of contracts simultaneously.

The trade-off is approximation error — the NN is not the exact Heston price. The key question this project addresses is whether that error is small enough to be acceptable in practice (spoiler: median relative error on ITM/ATM contracts is well below 1%).

---

## Part 1a — Monte Carlo Heston Simulator

**Notebook:** `1a_heston_monte_carlo_simulation.ipynb`

The Heston model describes the joint dynamics of the asset price $S_t$ and its instantaneous variance $Y_t$:

$$dS_t = r S_t \, dt + \sqrt{Y_t} S_t \, dW_t^1$$
$$dY_t = \kappa(\mu - Y_t) \, dt + \sigma \sqrt{Y_t} \, dW_t^2, \quad d\langle W^1, W^2\rangle_t = \rho \, dt$$

**Implementation details:**
- Full vectorized GPU simulation in PyTorch over batches of contracts
- Euler–Maruyama discretization with full truncation (variance clamped to 0)
- Correlated Brownian increments via Cholesky decomposition: $W^2 = \rho W^1 + \sqrt{1-\rho^2} Z$
- **Antithetic variates** for variance reduction — each draw uses both $Z$ and $-Z$ simultaneously
- **Feller condition** enforced during sampling ($2\kappa\mu > \sigma^2$) to keep the variance process strictly positive

**Dataset:**
| Split | Samples | MC paths | Steps |
|-------|---------|----------|-------|
| Train | 100,000 | 40,000   | 200   |
| Test  | 20,000  | 90,000   | 200   |

**Input space** — 9 features uniformly sampled:

| Feature | Symbol | Range |
|---------|--------|-------|
| Spot price | $S_0$ | [80, 120] |
| Initial variance | $Y_0$ | [0.01, 0.35] |
| Mean reversion speed | $\kappa$ | [0.5, 5.0] |
| Long-run variance | $\mu$ | [0.01, 0.25] |
| Vol of vol | $\sigma$ | [0.1, 1.0] |
| Risk-free rate | $r$ | [0.0, 0.1] |
| Correlation | $\rho$ | [-0.95, 0.0] |
| Strike | $K$ | [60, 140] |
| Maturity | $T$ | [0.1, 2.0] |

---

## Part 1b — MLP Surrogate Pricer

**Notebook:** `1b_heston_mlp_surrogate_pricer.ipynb`

Trains a feedforward neural network to approximate the Heston pricing function $\hat{P}(S_0, Y_0, \kappa, \mu, \sigma, r, \rho, K, T)$, replacing expensive Monte Carlo calls at inference time.

**Architecture (baseline):**
```
Input(9) → Linear(256) → BN → ReLU → Dropout(0.05)
         → Linear(256) → BN → ReLU → Dropout(0.05)
         → Linear(128) → BN → ReLU → Dropout(0.05)
         → Linear(64)  → BN → ReLU → Dropout(0.05)
         → Linear(1)
```

Three additional architectures are benchmarked: Small (256→128→64), Large (512→512→256→256→128→64), and VeryLarge (8 layers up to 512 units).

**Training setup:**
- Loss: Huber loss ($\delta = 0.5$) — robust to large-price outliers
- Optimizer: Adam (lr=1e-3, weight decay=1e-5)
- Scheduler: ReduceLROnPlateau (factor=0.5, patience=4)
- Early stopping: patience=15 epochs
- Input standardized using training-set statistics

**Evaluation:**
- Global MAE, RMSE, relative error
- Breakdown by contract type: OTM / ATM / ITM (moneyness $S_0/K$)
- Error vs each input feature (decile curves with 95% confidence intervals)
- Relative error reported only on contracts with price > 0.5 to avoid near-zero noise

---

## Part 2 — Neural Network Calibration

**Notebook:** `2_heston_calibration_neural_network.ipynb`

Uses the trained MLP surrogate as a **differentiable pricing oracle** to recover the 5 latent Heston parameters $(\kappa, \mu, \sigma, r, \rho)$ from an observed option price surface via gradient-based optimization.

**Setup:**
- Synthetic observed surface generated by high-accuracy MC (200k paths, 1000 steps) using known ground-truth parameters
- Surface: 8 maturities × 17 strikes = 136 contracts
- Known: $S_0 = 100$, $Y_0 = 0.04$; calibrate the 5 remaining parameters

**Calibration objective:**

$$\min_{\kappa, \mu, \sigma, r, \rho} \sum_{i,j} \left(\hat{P}_{NN}(S_0, Y_0, \kappa, \mu, \sigma, r, \rho, K_j, T_i) - P^{obs}_{ij}\right)^2$$

**Parameter constraints:**
Parameters are reparametrized via bounded sigmoid transforms to enforce domain bounds throughout optimization — no projection or clipping needed.

**Optimization:**
- Optimizer: Adam (lr=5e-3) with gradient clipping (norm=1.0)
- 6,000 iterations per run
- **Multi-start**: 10 random initializations to escape local minima; best run selected by final loss

**Evaluation:**
- Parameter recovery: calibrated vs true values (absolute and relative error)
- In-sample surface fit (same K, T grid used for calibration)
- Out-of-sample surface fit (unseen maturities and strikes)
- Convergence plots: loss curve and parameter trajectories across iterations

---

## Data Files

All generated artifacts are stored in the `data/` folder. They are produced sequentially — each notebook feeds into the next.

| File | Size | Produced by | Used by | Description |
|------|------|-------------|---------|-------------|
| `dataset_standard_mc.pt` | 4.6 MB | `1a` | — | 100k train + 20k test option prices generated by standard MC (no variance reduction). First version, superseded by the antithetic dataset. |
| `dataset_antithetic_mc.pt` | 4.6 MB | `1a` | `1b`, `2` | Same dataset regenerated with **antithetic variates** — lower MC variance, better label quality. This is the dataset used for all training. |
| `model_mlp_baseline.pt` | 454 KB | `1b` | — | Trained weights of the baseline MLP (architecture: 256→256→128→64). Includes preprocessing stats (`x_mean`, `x_std`). |
| `model_mlp_large.pt` | 2.0 MB | `1b` | `2` | Trained weights of the large MLP (architecture: 512→512→256→256→128→64). Best-performing model, used as the differentiable pricer in calibration. |
| `surface_synthetic_observed.pt` | 8 KB | `2` | `2` | Synthetic observed option price surface (8 maturities × 17 strikes) priced by high-accuracy MC (200k paths, 1000 steps) using known ground-truth Heston parameters. This is the calibration target. |
| `calibration_result_single_start.pt` | 203 KB | `2` | — | Output of a single-start gradient calibration run: best parameters, loss history, predicted surface. |
| `calibration_result_multistart.pt` | 186 KB | `2` | — | Output of the 10-start calibration: all runs + global best result selected by final loss. |

---

## Dependencies

```
torch
numpy
pandas
matplotlib
tqdm
```

All notebooks were developed on Google Colab with GPU acceleration. Data tensors are loaded from Google Drive (`heston_av_raw_dataset.pt`, `heston_mlp_model_large.pt`, `heston_observed_surface.pt`) — adjust paths if running locally.
