
import numpy as np

import mpnum as mp
import mpnum.povm as povm

n_sites = 8
ldim = 2
n_samples = 20
eps = 1e-10

rng = np.random.RandomState(seed=3770537836)
#psi = mp.MPArray.from_kron([np.array([1, -1j])] * n_sites)
psi = mp.random_mpa(n_sites, ldim, 2, rng)
psi /= mp.norm(psi)
rho = mp.mps_to_mpo(psi)

locpovm = povm.x_povm(ldim)
mpp = povm.MPPovm.from_local_povm(locpovm, n_sites)

s = mpp.sample(rng, rho, n_samples, direct_max_n_probab=-1, eps=eps)
print(s)
