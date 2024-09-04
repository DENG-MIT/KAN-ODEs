import matplotlib.pyplot as plt
import numpy as np

###This code generates plot 3(C). Rerunning the KANODE and MLP drivers to convergence and saving metrics here
###lets us investigate the trends.

kan_err=np.array([1.4e-4, 5.2e-5, 1.2e-4, 1.9e-5, 1.6e-5, 8.3e-7, 6.6e-7, 6.1e-7])
kan_size=np.array([64, 80, 96, 120, 144, 240, 480, 960])

MLP_err_2=np.array([4.7e-4, 4.14e-5, 1.6e-5])
MLP_size_2=np.array([52, 252, 502])

MLP_err_3=np.array([2.0e-4, 2.6e-4, 3.6e-5, 3.7e-5, 2.96e-5])
MLP_size_3=np.array([29, 57, 114, 162, 522])

N_2_x=[MLP_size_2[0]*0.9, MLP_size_2[-1]*1.5]
N_2_y=np.array([MLP_err_2[0]*1.7, (MLP_err_2[0]*1.7*(N_2_x[0]/N_2_x[1])**2)])

N_4_x=[kan_size[0]*0.8, kan_size[-3]*1.1]
N_4_y=np.array([kan_err[0], (kan_err[0]*(N_4_x[0]/N_4_x[-1])**4)])

plt.figure(figsize=[3.4, 2.8])
plt.loglog(kan_size, kan_err, '-^', label="KAN-ODE, d=2", color='maroon')
plt.loglog(MLP_size_2, MLP_err_2, '--o', label="MLP-NODE, d=2", color="midnightblue")
plt.loglog(MLP_size_3, MLP_err_3, '--o', label="MLP-NODE, d=3", color="mediumblue")
plt.loglog(N_2_x, N_2_y, linewidth=2.25, color="deepskyblue")
plt.loglog(N_4_x, N_4_y, linewidth=2.25, color='orangered')

plt.xlabel("# of parameters")
plt.ylabel("Converged loss")
plt.tight_layout()


plt.legend(bbox_to_anchor=(1, 1), loc=1, borderaxespad=0.2, borderpad=0.4, fontsize="8")
plt.savefig("trend_plot.png", dpi=200, facecolor="w", edgecolor="w", orientation="portrait")
