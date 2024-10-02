
During my research, I always assumed that neural fractional ODEs were a low-hanging fruit—so low-hanging that it should have fallen just months after the neural ODE paper. Given that the fractional adjoint method was theoretically well-understood (despite occasional errors), I initially assumed this concept had already been explored.

However, I was surprised to find that nobody I spoke with could point me to a single repository implementing the fractional adjoint method. So I decided to give it a try, and after struggling for days with convergence and the horrid limiting terminal conditions of the adjoint state, I eventually abandoned the effort. Despite this, I had a potential technique in mind—one that would be algorithm-agnostic and not just backpropagating through the numerical method—and I finally decided to bring that idea to life.

[https://github.com/AndresCenteno/deep_equilibrium_fodes](https://github.com/AndresCenteno/deep_equilibrium_fodes)

The core of the approach still relies on implicit differentiation, but instead of utilizing the adjoint method, it employs a well-known integral reformulation of the original problem. Essentially, the solution to the neural fractional ODE can be viewed as a fixed point of the integral equation, and the backward pass can be derived using the implicit function theorem. One could be more rigorous, make graphs, tables with benchmarks and make it into a paper. I have no time for that and wanted to just share it.

**Advantages:** This approach is parallelizable in time and proven to work. There are no discretization issues beyond the convergence of the forward pass, which we understand quite well.

**Disadvantages:** It requires more neural network evaluations if executed serially. However, with sufficient parallel processing power, it might outperform the adjoint method in terms of speed.
