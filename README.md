# Out-of-Domain Generalization in Dynamical Systems Reconstruction [[ICML 2024 Poster](https://icml.cc/virtual/2024/poster/32708)]
![](OODG_poster_2107.png "ICML Poster")

:warning: *Repository still WIP* :warning:

## 1 Setup
### 1.1 Environment setup
To rerun figure code, instantiate the Julia environment in the main repository directory.
```julia
julia> ]
(@v1.10) pkg> activate .
(OODG-in-DSR) pkg> instantiate
```
We recommend using the latest version of [Julia (>v1.10)](https://julialang.org/downloads/).

### 1.2 RNN/N-ODE/RC environments
If you want to train and evaluate models yourself, you should work with the respective environment. For example,
for RNNs, `cd` into the `RNNs` folder and instantiate the Julia environment.
#### RNNs
```bash
$ cd RNNs
$ julia
```
```julia
julia> ]
(@v1.10) pkg> activate .
(GTF) pkg> instantiate
```
#### RCs
```bash
$ cd ReservoirComputing
$ julia
```
```julia
julia> ]
(@v1.10) pkg> activate .
(ReservoirComputing) pkg> instantiate
```
#### N-ODEs
```bash
$ cd NeuralODEs
$ julia
```
```julia
julia> ]
(@v1.10) pkg> activate .
(NeuralODETraining) pkg> instantiate
```

## 2 Running figure code
To reproduce figure of the main paper, run the respective scripts in the folder `Figure1` to `Figure6`, e.g.
```bash
$ julia --project=. Figure1/figure1.jl
```
Note: `Figure2` is a Python Jupyter Notebook.

## 3 Running Model trainings and evaluation
### 3.1 RNNs
The code is based on https://github.com/DurstewitzLab/GTF-shPLRNN. To reproduce models for Fig. 3, run 
```bash
$ julia --project=RNNs RNNs/duffing_runs_fig3.jl -p X -r 50
```
where `X` is a placeholder for the number of parallel processes you want to spawn. Afterwards, `evaluate_duffing_runs.jl` can be used to evaluate the models:
```bash
$ julia --project=RNNs RNNs/evaluate_duffing_runs.jl
```
### 3.2 RCs
To train and evaluate RC models, run 
```bash
$ julia --project=ReservoirComputing ReservoirComputing/evaluate_duffing.jl
```
which trains and evaluates RCs based on grid search results in `gs_results_duffing.jld2`. You can also do your own grid search by running `grid_search.jl`.

### 3.3 N-ODEs
The structure follows the one for RNNs, that is,
```bash
$ julia --project=NeuralODEs NeuralODEs/duffing_runs_fig3.jl -p X -r 50
```
for training and
```bash
$ julia --project=NeuralODEs NeuralODEs/evaluate_duffing.jl -p X 
```
for evaluation (which in this case can also be accelerated by multiprocessing).

## 4 Miscellaneous
### 4.1 Measures
The measures introduced in the paper can be found in `ReconstructionMeasures/wasserstein_based_error.jl` for the statistical error and `ReconstructionMeasures/topological_error.jl` for the topological error, respectively. The grid of initial conditions on which the measures are evaluated can be generated and inspected using `ic_grid_duffing.jl` (and `ic_grid_lorenz_like.jl`):
```bash
$ julia --project=. ic_grid_duffing.jl
```
## 5 Citation
If you find the repository and/or paper helpful for your own research, please cite [our work](https://proceedings.mlr.press/v235/goring24a.html).
```
@InProceedings{pmlr-v235-goring24a,
  title = 	 {Out-of-Domain Generalization in Dynamical Systems Reconstruction},
  author =       {G\"{o}ring, Niclas Alexander and Hess, Florian and Brenner, Manuel and Monfared, Zahra and Durstewitz, Daniel},
  booktitle = 	 {Proceedings of the 41st International Conference on Machine Learning},
  pages = 	 {16071--16114},
  year = 	 {2024},
  editor = 	 {Salakhutdinov, Ruslan and Kolter, Zico and Heller, Katherine and Weller, Adrian and Oliver, Nuria and Scarlett, Jonathan and Berkenkamp, Felix},
  volume = 	 {235},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {21--27 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://raw.githubusercontent.com/mlresearch/v235/main/assets/goring24a/goring24a.pdf},
  url = 	 {https://proceedings.mlr.press/v235/goring24a.html},
  abstract = 	 {In science we are interested in finding the governing equations, the dynamical rules, underlying empirical phenomena. While traditionally scientific models are derived through cycles of human insight and experimentation, recently deep learning (DL) techniques have been advanced to reconstruct dynamical systems (DS) directly from time series data. State-of-the-art dynamical systems reconstruction (DSR) methods show promise in capturing invariant and long-term properties of observed DS, but their ability to generalize to unobserved domains remains an open challenge. Yet, this is a crucial property we would expect from any viable scientific theory. In this work, we provide a formal framework that addresses generalization in DSR. We explain why and how out-of-domain (OOD) generalization (OODG) in DSR profoundly differs from OODG considered elsewhere in machine learning. We introduce mathematical notions based on topological concepts and ergodic theory to formalize the idea of learnability of a DSR model. We formally prove that black-box DL techniques, without adequate structural priors, generally will not be able to learn a generalizing DSR model. We also show this empirically, considering major classes of DSR algorithms proposed so far, and illustrate where and why they fail to generalize across the whole phase space. Our study provides the first comprehensive mathematical treatment of OODG in DSR, and gives a deeper conceptual understanding of where the fundamental problems in OODG lie and how they could possibly be addressed in practice.}
}
```
## 6 Funding
This work was funded by the German Research Foundation (DFG) within Germany’s Excellence Strategy EXC 2181/1 – 390900948 (STRUCTURES) and by DFG grant Du354/15-1 to DD.


