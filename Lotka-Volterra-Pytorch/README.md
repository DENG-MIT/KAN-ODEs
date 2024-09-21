# Auxillary Pytorch code

predatory_prey.py is the driver that runs the Lotka-Volterra training in PyTorch. Results are saved into /plots/. Plots from a test run by the authors are included in this repository.

predator_prey_adjoint.py uses the adjoint method, which we found to be slightly slower given the small KAN-ODE size studied here.

This implementation relies on the efficient-kan python package from https://github.com/Blealtan/efficient-kan, which is Ref. [44] in the CMAME manuscript.

**This implementation is many times slower (~50x according to our tests) than the Julia implementation for the Lotka-Volterra case, and appears to converge to a poorer result with larger overfitting. We strongly recommend the julia implementation for KAN-ODE users, although we include this Python code for any users interested in developing KAN-ODEs in Python further.**

As a starting point for any future KAN-ODE development in Python, we refer to the following repositories which discuss KAN speed-ups and other improvements in Python:

https://github.com/AthanasiosDelis/faster-kan

https://github.com/mintisan/awesome-kan