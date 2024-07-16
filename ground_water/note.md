# Note for solving transition diffusion equation

This is the Note for recording my thinking and some numerical results for the groundwater flow equation

## Problem Setting(Groundwater flow equation)

Control equation:
$$
\begin{equation}
\left\{
\begin{aligned}
    &\frac{\partial u}{\partial t} - \nabla\cdot F = - f(\bold{x}, u, t); &(\bold{x}, t) \in \Omega \times (0, T] \\
    &F =  \Lambda(\bold{x}, u, t)\nabla u
\end{aligned}\right.
\end{equation}
$$
with the initial and boundary conditions:
$$
\begin{equation}
\begin{aligned}
    &u(\bold{x}, 0) = g_0(\bold{x});    &\bold{x} \in \Omega \\
    &u(\bold{x}, t) = g_D(\bold{x}, t); &(\bold{x}, t) \in \Gamma_D\times (0, T] \\
    &\Lambda\nabla u(\bold{x}, t) \cdot \bold{n}(\bold{x}) = g_N(\bold{x}); &(\bold{x}, t) \in \Gamma_N\times (0, T] \\
\end{aligned}
\end{equation}
$$

The GroundWater equation is a diffusion equation which can be solved by the Finite Volume method.
We discretized the equation on a cell-centered uniform rectangular mesh $\mathcal{M} = \{\bold{x}\}_{ij=1}^{N,M}$ whose spatial size and time step are $\Delta x$ and $\Delta t$ respectively.
The node $\bold{x}_{ij}$ is the center of the $(i,j)$ elements in the mesh.

The Gauss theorem is employed to transform the integral of the divergence into the normal fluxes on the boundaries after integrating the diffusion equation over a element.
A well-known five-point finite volume scheme is proposed to discretize the equation on the mesh $\mathcal{M}$.
$$
\int_{E_{ij}} \frac{\partial u}{\partial t} d\bold{x} - \int_{E_{ij}} \nabla \cdot F d\bold{x} = -\int_{E_{ij}} f(x, u, t)d\bold{x}
$$
$$
|E_{ij}| \frac{u^{n+1}_{ij} - u^{n}_{ij}}{\Delta t} - \oint_{E_{ij}} \Lambda \frac{\partial u}{\partial \bold{n}} ds = -|E_{ij}|f(x, u, t)
$$

$$
\Delta x^2 \frac{u^{n+1}_{ij} - u^{n}_{ij}}{\Delta t} + \sum_{l=1}^4 F_{ij;l}^{n+1} = -\Delta x^2 f(x_{ij}, u^{n+1}_{ij}, t^{n+1})
$$
where the discrete flux for the left side $l=1$ is given by:
$$
\begin{equation}
F^{n+1}_{ij;1} = \Lambda_{ij;1}^{n+1}\frac{u^{n+1}_{i-1,j} - u^{n+1}_{ij}}{2\Delta x} 
\end{equation}
$$
$$
\begin{equation}
\Lambda_{ij;1}^{n+1} = \frac{2\Lambda_{ij}^{n+1}\Lambda_{i-1,j}^{n+1}}{\Lambda_{ij}^{n+1} + \Lambda_{i-1,j}^{n+1}}
\end{equation}
$$

This scheme is shown as follows:
$$
\begin{equation}
\frac{u^{n+1}_{ij} - u^{n}_{ij}}{\Delta t} + \frac{1}{\Delta x^2}\sum_{l=1}^m F^{n+1}_{ij;l} = -f^{n+1}_{ij}
\end{equation}
$$

All the fluxes for the four edges of a rectangular element can be calculated as follows:
$$
\begin{equation}\left\{
\begin{aligned}
F^{n+1}_{ij;1}\Delta x = 2(\frac{1}{\Lambda^s_{i-1,j}} + \frac{1}{\Lambda^s_{ij}})^{-1}(u^{n+1}_{i-1,j} - u^{n+1}_{ij})\\
F^{n+1}_{ij;2}\Delta x = 2(\frac{1}{\Lambda^s_{i,j-1}} + \frac{1}{\Lambda^s_{ij}})^{-1}(u^{n+1}_{i,j-1} - u^{n+1}_{ij})\\
F^{n+1}_{ij;3}\Delta x = 2(\frac{1}{\Lambda^s_{i+1,j}} + \frac{1}{\Lambda^s_{ij}})^{-1}(u^{n+1}_{i+1,j} - u^{n+1}_{ij})\\
F^{n+1}_{ij;4}\Delta x = 2(\frac{1}{\Lambda^s_{i,j+1}} + \frac{1}{\Lambda^s_{ij}})^{-1}(u^{n+1}_{i,j+1} - u^{n+1}_{ij})\\
\end{aligned}\right.
\end{equation}
$$
Therefore, the final discrete equation is formulated as follows:

$$
\begin{equation}
\begin{aligned}
\frac{\Delta x^2}{2}(\frac{u^{n+1}_{ij} - u^{n}_{ij}}{\Delta t} + f^{n+1}_{ij}) = ~& (\frac{1}{\Lambda^s_{i-1,j}} - \frac{1}{\Lambda^s_{ij}})^{-1}(u^{n+1}_{i-1,j} - u^{n+1}_{ij}) + (\frac{1}{\Lambda^s_{i+1,j}} - \frac{1}{\Lambda^s_{ij}})^{-1}(u^{n+1}_{i+1,j} - u^{n+1}_{ij}) \\
&(\frac{1}{\Lambda^s_{i,j-1}} - \frac{1}{\Lambda^s_{ij}})^{-1}(u^{n+1}_{i,j-1} - u^{n+1}_{ij}) + (\frac{1}{\Lambda^s_{i,j+1}} - \frac{1}{\Lambda^s_{ij}})^{-1}(u^{n+1}_{i,j+1} - u^{n+1}_{ij})
\end{aligned}
\end{equation}
$$


The figure shown above is used to illustrate the idea, there are $(4\times 4)$ blue nodes in the figure and the Dirichlet boundary nodes drawn in red nodes are padded onto the prediction and input.
The edges colorized as green are the left edges of all elements in this mesh with size of $(6\times 6)$.

1. Make a prediction $\bold{h}^n_\theta = \mathcal{N}_\theta(\bold{h}^{n-1})$.
2. Compute the coefficient $\Lambda(h_\theta)$ and inverse it $\widetilde{\Lambda} = \frac{1}{\Lambda}$. Padding $0$ onto the left side of $\widetilde{\Lambda}$.
3. Padding the Dirichlet boundary $g_D$ on the left side of the prediction and input. The padded prediction has the shape of $(N+1\times N)$.

 > If the boundary condition on the left side is the Neumann boundary condition, the value $g_N$ has no needs to be padded. Instead, it should be add onto the flux directly with $g_N \Delta x$.
  <div style="text-align: center;">
    <img src="kappa.png" alt="" />
  </div>
  
3.Compute the flux for all the left edges $\{\sigma_{ij}^1\}_{i,j=1}^N$ with the help of a convolution operator whose kernel is $[-1, 1]$, denoted by $\mathcal{K}_h$.
And another convolution operator whose kernel is $[1, 1]$ denoted by $\mathcal{K}_
\Lambda$. The equation $(3)$ is calculated through these two convolutions and can be formulated as follows:

$$
\begin{equation}
\bold{F}_{left}^n =\frac{\mathcal{K}_h * h}{\mathcal{K}_\Lambda * \widetilde{K}}
\end{equation}
$$

Another scheme for Picard's iteration from **An efficient parallel iteration algorithm for nonlinear diffusion equations with time extrapolation techniques and the Jacobi explicit scheme**. 
