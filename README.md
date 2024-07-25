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
If you find the repository and/or paper helpful for your own research, please cite our work (will be replaced by PMLR cit. once published).
```
@misc{göring2024outofdomaingeneralizationdynamicalsystems,
      title={Out-of-Domain Generalization in Dynamical Systems Reconstruction}, 
      author={Niclas Göring and Florian Hess and Manuel Brenner and Zahra Monfared and Daniel Durstewitz},
      year={2024},
      eprint={2402.18377},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2402.18377}, 
}
```
## 6 Funding
This work was funded by the German Research Foundation (DFG) within Germany’s Excellence Strategy EXC 2181/1 – 390900948 (STRUCTURES) and by DFG grant Du354/15-1 to DD.


