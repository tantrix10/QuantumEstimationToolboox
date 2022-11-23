# QFIM_Genetic_optimisation

This work uses a genetic algoirthm to optimise the quantum Fisher informatin for magnetic field estimation.
The code base is split into 4 main sections:
- optimiser:
  - this contains the genetic algorithm that perfroms that optimisation
- cost_func:
  - this is where the various possible cost functins are stored
  - current cost functions implemented: QFI, CFI, estimator varience
- dynamics:
  - contains the code defining the evolution of the pure input state to evolved state, including Kraus noise.
- states_measurements:
  - a library of useful input states and measurements (useful for CFI/estimator cost functions)

## Dynamics 

### Magnetic field estimation


In this work we are interested in the estimation of the 3 components of a magnetic field, given by Hamiltonian,
$$
H(\varphi) = \varphi_1(\sigma_x\otimes \mathbb{I}+ \mathbb{I}\otimes \sigma_x)+\varphi_2(\sigma_y\otimes \mathbb{I}+ \mathbb{I}\otimes \sigma_y)+\varphi_3(\sigma_z\otimes \mathbb{I}+ \mathbb{I}\otimes \sigma_z),
$$
where we introduced the standard Pauli matrices
$$
\sigma_z = \begin{bmatrix}
1 & 0 \\
0 & -1
\end{bmatrix} \quad
\sigma_x = \begin{bmatrix}
0 & 1 \\
1 & 0
\end{bmatrix} \quad
\sigma_y = \begin{bmatrix}
0 & i \\
-i & 0
\end{bmatrix}.
$$
With subsequent evolution,
$$
\rho{\psi_{\varphi, \gamma}} =\sum_i E_i e^{-i H }\bra{\psi_0}\ket{\psi_0}e^{i H } E_i^\dagger,
$$
where $\ket{\psi_0}$ is some pure input state and {E_i} is a valid set of Kraus operators corresponding to some noisy evolution. 

It is the pure input state, $$\ket{\psi_0}$$ that will be optimised by the genetic algorithm.


## Cost functions

### Quantum Fisher Information


Let us begin with $\lambda$, the single parameter we wish to estimate. With $p(x_i,\lambda)$ being the probability distribution of the set of measurements $\{x_i\}$ that have been taken and are being used to estimate $\lambda$. The classical Cram\'er-Rao bound states that the variance of an unbiased estimator, $\hat{\lambda}$ of $\lambda$ is given by the reciprocal of the classical Fisher information,
$$
Var(\hat{\lambda})\geq \frac{1}{M\mathcal{F}(\lambda)},
$$
where $M$ is the number of measurements taken. The classical Fisher information $\mathcal{I}(\lambda)$ is defined as \cite{Banerjee2016QuantumQubit},
\textcolor{red}{need to check this equation in the tex}
$$
\mathcal{F}(\lambda)&=\text{E}\left[\left(\partial_\lambda \text{ln}(p_i(x_i,\lambda))\right)^2 \right]\\
&=\sum_i p_i(x_i,\lambda)\left(\frac{\partial \text{ln}(p_i(x_i,\lambda))}{\partial\lambda}\right)^2.
$$


\noindent We can further lower bound the varience of the estimate $\hat{\lambda}$ by the so called quantum Cram\'er-Rao bound (QCRB), given by \cite{Yue2016InvertibleQubit},
$$
Var(\hat{\lambda})\geq \frac{1}{M\mathcal{F}(\lambda)} \geq \frac{1}{M\mathcal{I}(\lambda)},
$$
where $\mathcal{I}(\lambda)$ denotes the quantum Fisher information (QFI). Which is defined to be,
$$
\mathcal{I}(\lambda) = \sum_x \frac{\Tr[\rho_\lambda\Pi_x L_\lambda]^2}{\Tr[\rho_\lambda \Pi_x]},
$$
where $\{\Pi_x \} $ is the set of POVMs. The POVMs give rise to the probability distribution $p(x|\lambda)=\Tr[\Pi_x\rho_\lambda]$ and $L_\lambda$ denotes the Symmetric Logarithmic Derivative (\ac{SLD}). The SLD is a self adjoint operator satisfying the equation,
$$
\partial_\lambda\rho_\lambda = \frac{L\rho+\rho L}{2},
$$
where, $\rho_\lambda$ denotes the state which has the parameter encoded on it and $\Tr$ is the trace operator of a matrix given by $\sum_j a_{j,j}$ where $a_{j,j}$ are the diagonal elements of a square matrix.
We are also able to write the QFI in a simpler form,
$$
\mathcal{I}(\lambda) = \Tr[\rho_\lambda L_\lambda^2].
$$
\noindent If we are restricted to pure states we are able to use a simpler expression,
$$
\mathcal{I}(\lambda)&=4\left(\frac{\partial}{\partial \lambda}\bra{\psi}U(\lambda,t),\frac{\partial}{\partial \lambda}U(\lambda,t)\ket{\psi} -\left(\frac{\partial}{\partial \lambda}\braket{\psi U(\lambda,t),\psi}\right)^2 \right),\\
&=4t^2\left(\bra{\psi}H^2\ket{\psi}- \left(\bra{\psi}H\ket{\psi}\right)^2\right).
$$
This step is justified as the unitary evolution Hamiltonian $H(\lambda)=\lambda H$ for the observable part of the overall Hamiltonian. Here, $\ket{\psi}$ is the probe state used to estimate $\lambda$ and $U(\lambda,t)=e^{iH(\lambda)t}$ is the unitary evolution the probe state undergoes. Now, if we have the Greenberger–Horne–Zeilinger (\ac{GHZ}) state,
$$
\ket{\psi}_{GHZ}=\frac{\ket{0}^{\otimes N}+\ket{1}^{\otimes N}}{\sqrt{2}},
$$
taking particular note that $\ket{0},\ket{1}$, correspond to the eigenstates for the maximum and minimum eigenvalues of the observable Hamiltonian then $F(\lambda)\propto t^2N^2 $. This is the quantum speed up we are trying to achieve. Classically, if we have $N$ probe states then the maximum scaling achievable is $\mathcal{I}(\lambda)\propto N$, this is the classical scaling. Whereas the so called Heisenberg scaling is $\mathcal{F}(\lambda)\propto N^2$. We are also able to use the single qubit formulation of the quantum Fisher information \cite{Banerjee2016QuantumQubit},
$$
\mathcal{I}(\lambda)=\frac{\braket{\underline{v}(\lambda),\partial_{\lambda} \underline{v}(\lambda)}}{\mathbb{I} - \Vert \underline{v}(\lambda) \Vert ^2}+\left\Vert \frac{\partial}{\partial \lambda}\underline{v}(\lambda) \right\Vert ^2,
$$

\noindent where $\underline{v}(\lambda)$, is the Bloch vector that has undergone the evolution,
$$
\rho = \frac{1}{2}(\mathbb{1}+\underline{v}(\lambda) \cdot \sigma_i).
$$

## Optimser

### Genetic Algorithm

The genetic agorithm treats the pure input state as the "gene". Each gene has an assosiated cost function value, here we will assume qfi for description. For each iteration, we generate a mating pool. The probability of being seleted to enter the mating pool is proportional to how 'good' its cost function is. Once we have a mating pool, pairs (parents) are randomly selected from the pool. A random splice point is then chosen, a new 'child' is then built up from everything below the first parent at the index and above from the other. A round of mutations then takes place. Finally the 'best state' is always retained across iterations. This continues until the defined number of iterations.

<img src="https://raw.githubusercontent.com/tantrix10/thesis/master/optimal_state_chap/figures/genetic.pdf?token=AE7QLCCRKNEKMFINBAXJDFC7Q34YQ"
     alt="Genetic Algorithm"
     style="float: left; margin-right: 10px;" />


## Instalation & running

### Instalation
Clone from github with:
> git clone https://github.com/tantrix10/QFIM_Genetic_optimisation.git

and install with:

> pip install -r requirements.txt

n.b. it only requires: 
- matplotlib
- numpy
- qutip
- scipy
- pytest


which are fairly standard quantum information libraries to have already installed if you are not working in a venv (which is recommended, if you are unfamiliar with venvs, check out *[this](https://packaging.python.org/tutorials/installing-packages/#creating-virtual-environments)*).

Once this is cloned, feel free to move into the cloned directory and run 
```python
> .../QFIM_GENETIC_OPTIMISATION: ~$ pytest
```
in terminal in order to make sure everything is installed properly and all tests pass as expected :) 

## Example code 
```python
from optimiser import opt_state
from cost_func import qfi
from dynamics import final_ state, SC_estim_ham
from states_measurements import til_state
from qutip import sigmax, sigmay, sigmaz

paulis = [sigax(), sigmay(), sigmaz()]

n = 2
nos = 10
gamma = 0
itter = 500
alpha = [0,0,0]
hams = [SC_estim_ham(pauli) for pauli in paulis]

def opt_state(
    n,
    itter,
    qfi,
    final_state,
    init_state,
    alpha,
    hams,
    nos,
    gamma,
    mutatation_rate=None,
    verbose=True,
    save=True,
)
```
# Issues

For any issues, comments, bugs, general thoughts and feelings, I am the sole author/maintainer of this code base so just put a github issue in and I will get back to you ASAP. Similarly for new feature pull requests.