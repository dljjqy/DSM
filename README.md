# A novel deep convolutional surrogate model with incomplete solve loss for parameterized steady-state diffusion problems

Code repository for the paper:
[A novel deep convolutional surrogate model with incomplete solve loss for parameterized steady-state diffusion problems](https://doi.org/10.1016/j.jcp.2025.114132).

## Usage

1. Traditional [FVM](https://doi.org/10.1016/j.camwa.2022.11.023) for generating Labeled data in FVM folder. The original paper and numerical results are listed in __note__ folder.

2. Ablation study are listed in demo folder.

3. Three applications are shown in folder __fit_heat__, __fit_multicof__ and __fit_nlinear__. Code for train and test are implemented using Python script and Jupyter notebook separately.

4. The code for the 3D numerical case is listed in __fit_3d__ folder.

## Citation

If you  find the idea or code of this paper useful for your research, please consider citing us:

```bibtex
@article{JIA2025114132,
title = {A novel deep convolutional surrogate model with incomplete solve loss for parameterized steady-state diffusion problems},
journal = {Journal of Computational Physics},
volume = {537},
pages = {114132},
year = {2025},
issn = {0021-9991},
}
```

## Tips

This paper focuses on constructing deep surrogate models for solving reaction–diffusion problems.   
It is easy to observe that when training neural networks using traditional numerical schemes, the following inequality holds:

$$
\|u - u_\theta \| \leq \|u - u^h\| + \|u^h - u_\theta\|,
$$

where $u$, $u^h$, and $u_\theta$ denote the exact solution, the numerical solution, and the neural network prediction, respectively.  Therefore, in this paper, we only focus on the optimization error that we are able to reduce, without expecting the surrogate model to surpass the accuracy of traditional numerical schemes.

- The first term on the right-hand side, $\|u - u^h\|$, comes from the traditional numerical discretization—it could be understood as the intrinsic error carried by the data.  
- The second term, $\|u^h - u_\theta\|$, is the optimization error introduced during the training of the deep surrogate model and is the only part we can control by tuning the training process.  
- Then, using the __incomplete iterative generator__ to generate pseudo-label data for a robustly, unsupervised training process, as it implicitly enhances the dataset while avoiding the introduction of linear systems during backpropagation.
- For the nonlinear problem, if you use the Picard iteration and need a $u_0$ to start the training, we recommond use the $u_0$ which you would use for the traditional numerical schemes. And for the nonlinear problem defined in this paper, we use the __ALL ONES__ matrix to initialize the training.


Overall, for different PDEs, when constructing deep surrogate models using the proposed methodology,

- The first step is to select an appropriate traditional numerical scheme.  In simple terms, we need to choose an *excellent teacher* for the neural network.  

- Next, depending on the characteristics of the problem, such as multiscale behavior or high-frequency components, we should select different network architectures to reduce the optimization error as much as possible; in other words, we need to choose an *excellent student*.  

__Finally, patience is crucial during training.__  
Except for the simple examples discussed in the `__demo__` folder, all other application problems in this codebase require extensive computation time for training and hyperparameter tuning.  This is inherently determined by the complexity of parametric PDEs. While we aim to reduce the optimization error as much as possible, we must also prevent overfitting. This often requires sampling a large number of parameter instances to support the training of the neural network, which naturally leads to a large number of optimization steps, in other words, significant computational time.
