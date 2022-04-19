# Continuous Constrained Optimization for Learning DAGs

This repository contains an implementation of the structure learning methods described in ["On the Convergence of Continuous Constrained Optimization for Structure Learning"](https://arxiv.org/abs/2011.11150). 

If you find it useful, please consider citing:
```bibtex
@inproceedings{Ng2022convergence,
  author = {Ng, Ignavier and Lachapelle, Sébastien and Ke, Nan Rosemary and Lacoste-Julien, Simon and Zhang, Kun},
  title = {On the Convergence of Continuous Constrained Optimization for Structure Learning},
  booktitle = {International Conference on Artificial Intelligence and Statistics},
  year = {2022},
}
```

## Requirements

- Python 3.6+
- `numpy`
- `scipy`
- `python-igraph`
- `torch`

## Running NOTEARS(-MLP) with QPM and ALM
- To use quadratic penalty method, set `opt_type` to `qpm`.
- To use augmented Lagrangian method, set `opt_type` to `alm`.
- See [examples/demo_linear.ipynb](https://github.com/ignavierng/notears-convergence/blob/master/examples/demo_linear.ipynb) and [examples/demo_nonlinear.ipynb](https://github.com/ignavierng/notears-convergence/blob/master/examples/demo_nonlinear.ipynb) for a demo in the linear and nonlinear cases, respectively.


## Acknowledgments
- Most of the code is obtained and modified from the implementation of [NOTEARS](https://github.com/xunzheng/notears), and we are grateful to the authors of NOTEARS for releasing their code.