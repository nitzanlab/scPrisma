import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from numba import prange

from scipy.stats import entropy
from algorithms import *
from data_gen import *
from datasets import *

#A = simulate_window_linear_2(1000,100,w=0.3)
#plot_covariance_matrix(A, "Linear signal")
#a=1/0
reconstruction_cyclic()
A = simulate_cyclic_random_k(30, 20,k_down=1, k_up=5)#, w=0.3)
plt.imshow(A)
plt.colorbar()
plt.show()
a=1/0
ranged_pca_2d(A,color=A[:,0] , title='')
ranged_pca_2d(A,color=A[:,1] , title='')
ranged_pca_2d(A,color=A[:,2] , title='')
ranged_pca_2d(A,color=np.clip(np.random.normal(0,1,A.shape[0]),0,1), title='')

a=1/0
noise = np.random.normal(0,  0.15, A.shape)
A = A+noise
print("start")
enhancement_cyclic_per_gene(A)
print("done")
enhancement_cyclic(A)
print("done")
a=1/0
for i in range(10):
    noise = np.random.normal(0,0.15,(100,100))
    print((np.mean(consistency_check(noise , nexp=10))))
a=1/0
mean_list= []
noise_var = []
for i in range(15):
    A = simulate_spatial_cyclic(100, 100, w=0.3)
    noise = np.random.normal(0,i*0.15,A.shape)
    A+=noise
    mean_list.append(np.mean(consistency_check(A , nexp=10)))
    noise_var.append(i*0.1)
plt.plot(noise_var,mean_list)
plt.show()
a=1/0
a = read_all_scn_no_obs()
def calculate_mean_normalized_entropy(E,n):
    mean=0
    for i in range(n):
        mean+= entropy(E[i,:])/np.log(n)
    return mean/n
tmp_noise_list = [
]
tmp_entropy = []

for i in range(25):
    A = simulate_spatial_cyclic(1000, 500, w=0.3)
    noise = np.random.normal(0,i*0.1,(500,1000))
    tmp_noise_list.append(i*0.1)
    B = A + noise
    E , E_recon , final_loss = reconstruction_cyclic(B, iterNum=100 , verbose=False , final_loss=True)
    tmp_entropy.append(final_loss)
    print(final_loss)
plt.plot(tmp_noise_list,tmp_entropy)
plt.show()
plt.plot(np.log(np.array(tmp_noise_list) +1),np.log(np.array(tmp_entropy) +1))
plt.show()
a=1/0
A = simulate_spatial_cyclic(100,50,w=0.3)
E , E_recon  = reconstruction_cyclic(A)
print(calculate_mean_normalized_entropy(E,50))
a=1/0
print(entropy([0.5,0.5]))
a=1/0
print(generate_spectral_matrix(10,0.9))
a=1/0
A= np.random.normal(3,1,(50,50))
F = reconstruction_cyclic(A)
F = filter_linear_genes(A)
