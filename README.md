# TenSAT

### Dependency
```
pytorch
sklearn
matplotlib
PyMiniSolvers
```

### Data Generation
To generate random SAT problem, run:
```
sh gen_data.sh
```

### Training
To train a SAT solver, you could try:
```
sh run_train.sh
```
In `run_train_sh`, the `baseModel` has two options: `neurosat` for NeuroSAT and `nnsat` for TenSAT. 

The `normal` can be set as `1` for using normalized adjacency matrix (GGCN).

| model | baseModel | normal |
|  ----  | ----  | ----  |
| NeuroSAT | neurosat | 0 |
| GGCN | neurosat | 1 |
| TenSAT(NeuroSAT) | nnsat | 0 |
| TenSAT(GGCN) | nnsat | 1 |

### Testing
To test models, run:
```
sh run_test.sh
```