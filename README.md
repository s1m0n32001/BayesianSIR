# BayesianSIR
Covid data analysis

<!-- ![plot](./figures/SIR.png) -->

In this work we aim to recreate the results from the stochastic SIR model with change points, proposed in [this paper](https://www.nature.com/articles/s41598-022-25473-y#MOESM1). We are going to test it both on a simulated time series and on the epidemiological dataset from the Omicron wave of 2022 in Singapore. 

We start from the equations of the stochastic SIR model:

$$ 
\begin{cases}
  \Delta I_t \sim \text{Binomial}(S_{t-1}, 1-exp(-\beta_tP_{t-1})) \\
  \Delta R_t \sim \text{Binomial}(I_{t-1}, \gamma_t) \\
  S_t = S_{t-1} - \Delta I_t \\
  I_t = I_{t-1} + \Delta I_t - \Delta R_t \\
  R_t = R_{t-1} + \Delta R_t \\
\end{cases}
$$

Where the transmission and removal rates $\beta = (\beta_1, ..., \beta_T), \gamma = (\gamma_1, ..., \gamma_T)$ depend on time. 
They are drawn from probability distributions whose characteristic parameter ($b$ or $r$) changes as the epidemic enters a different stage. Thus, the *change points* correspond to a shift in the average transmission/removal rates. This is summarized in the vector 

$$\delta_t = 
\begin{cases}
1 \text{ if $t$ is a change point}\\
0 \text{ otherwise}
\end{cases} $$

Starting from the time series $(\Delta I_t, \Delta R_t)_{t=1,...,T}$ with the initial conditions $(S_0,I_0,R_0)$, our goal is to infere $\delta, b,r, \beta, \gamma$.

The posterior probability is sampled through a Markov Chain Monte Carlo algorithm, which looks like this:

1. propose a new $\delta$: delete, add or swap a $1$.
2. accept it with probability $p = \text{min}(1,\pi_{MH})$
3. propose 

So overall the flowchart is 
  for g in range (G):
    propose delta 
      -> accept -> change b,r,beta,gamma
      -> refuse -> keep values
  




