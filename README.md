# Solving PDEs: Classical Methods vs Physics-Informed Neural Networks

This is the follow-up to my ODE project, extended to partial differential equations — equations where the unknown changes across space and time together, not just time. Same idea as before: solve it the classical way, solve it with a neural network trained on the physics instead of labeled data, and compare them honestly.

The classical method here is the Method of Lines — you discretize space into a grid, which turns the PDE into a big system of ODEs (one per grid point), and then you solve that with the same kind of ODE solver from the last project. It's a real, commonly used technique, not a simplification for this notebook.

The neural network approach is a Physics-Informed Neural Network (PINN), same as before but now taking two inputs, position and time, instead of just time.

## Two examples

**Heat equation** — a rod heating up and cooling down, with a known exact formula to check against. Both methods do well here. The Method of Lines is close to exact, and the PINN gets the right decaying shape with a bit more error, similar to what happened in the ODE project's linear example.

**Burgers' equation** — this one develops a steep, almost vertical front in the middle of the domain as time goes on, kind of like a shock wave. It's actually the same benchmark problem used in the original PINN research paper, so it felt like the right nonlinear test case to include. The PINN gets the overall shape right almost everywhere, but its biggest errors show up right at that steep front — which lines up with what's already known about basic PINNs struggling near sharp features.

## What's in the notebook

- The equations explained in plain language before any code
- Method of Lines solver for both problems, using `scipy.integrate.solve_ivp`
- A PINN built in PyTorch for each problem, trained only on the physics equation plus initial/boundary conditions — no labeled solution data
- Error plots comparing both methods, including a heatmap of where the PINN's error is worst on the Burgers' example
- A closing section on when a PINN is actually worth using over a classical solver

## Why I kept the PINN's weak spot in

The Burgers' example is where the PINN struggles most, and I didn't tune it until that error disappeared. A basic PINN like this spends equal effort across the whole domain, so a small sharp feature in the middle of an otherwise easy problem is genuinely hard for it — that's a known, documented limitation, not something specific to this notebook. Real research-grade PINNs get around this with tricks like sampling more points near the sharp region, but that's a step beyond what's shown here. I think it's more useful to see this honestly than to quietly tune it away.

## Running it

Open in Colab and run all cells. PyTorch installs in the first cell. Training both PINNs takes a while — the Burgers' one in particular runs for 10,000 epochs, so give it a few minutes.

[Open in Colab](https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO/blob/main/pde_solver_classical_vs_pinn.ipynb)
