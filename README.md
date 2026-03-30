# Robust Reinforcement Learning for Market Making under Regime Uncertainty: Numerical Evidence from Multi-Regime Training

This repository provides `mbt_gym` (model-based limit order book Gym environments) and the accompanying numerical study showing that multi-regime training improves out-of-sample return-side robustness under unseen market regimes.

## mbt_gym

`mbt_gym` provides a suite of Gym environments for training reinforcement learning (RL) agents on model-based high-frequency trading problems, such as market making and optimal execution. The environments are composable and optimized for efficient (vectorized) rollout generation.

This repository also includes a numerical study on *robust RL under market regime uncertainty*, where the trading environment parameters change between training and testing regimes.

## Regime Uncertainty (Numerical Study)

In the experiments, the market is parameterized by a regime vector:

`m = (sigma, A, k)`

- `sigma`: mid-price volatility
- `A`: baseline order-arrival intensity (liquidity)
- `k`: fill probability decay sensitivity w.r.t. quote distance

Two training settings are compared:

- **Single-regime PPO**: PPO trained on a single seen regime (e.g., `R2`)
- **Multi-regime PPO**: PPO trained with episode-level random regime sampling over a set of seen regimes (`R1`-`R6`)

Unseen test regimes:

- `U1`: interpolation-type unseen market
- `U2`: stress-type unseen market

Evaluation emphasizes return-side robustness using metrics such as `Mean PnL`, `CVaR10%`, and `CVaR30%`, while also reporting terminal inventory diagnostics.

## Repository Layout

- `mbt_gym/`: environment implementations, dynamics, rewards, wrappers, and agents.
- `experiment/`: notebooks and scripts to reproduce the numerical experiments (see `Template_Experiments.ipynb`).
- `requirements.txt`: dependencies for running the experiments (exported from the `RL2` conda environment).

## Setup

```bash
pip install -r requirements.txt
```

## Reproduce the Experiments

1. Open `experiment/Template_Experiments.ipynb`
2. Locate `REGIME_PARAMS` in the notebook to view the regime grid (`R1`-`R6`, `U1`, `U2`)
3. Run the training/evaluation cells and compare baselines:
   - Fixed Spread
   - Avellaneda-Stoikov
   - Single-regime PPO (e.g., trained only on `R2`)
   - Multi-regime PPO (uniform sampling across `R1`-`R6`)

## Citing `mbt_gym`

When using `mbt_gym`, please cite our [ACM ICAIF 2023 paper](https://arxiv.org/abs/2209.07823) with:

```bibtex
@inproceedings{JeromeSSH23,
  author       = {Joseph Jerome and
                  Leandro S{\'{a}}nchez{-}Betancourt and
                  Rahul Savani and
                  Martin Herdegen},
  title        = {Mbt-gym: Reinforcement learning for model-based limit order book trading},
  booktitle    = {4th {ACM} International Conference on {AI} in Finance, {ICAIF} 2023,
                  Brooklyn, NY, USA, November 27-29, 2023},
  pages        = {619--627},
  publisher    = {{ACM}},
  year         = {2023},
  url          = {https://doi.org/10.1145/3604237.3626873},
  doi          = {10.1145/3604237.3626873},
  note         = {arXiv preprint arXiv:2209.07823}
}
```

For the robust regime-uncertainty study motivating the `experiment/` notebooks, see:
`Robust Reinforcement Learning for Market Making under Regime Uncertainty: Numerical Evidence from Multi-Regime Training`.
Code (as referenced in the manuscript): https://github.com/tang-lin0203/MultiRegimeInMbtGym
