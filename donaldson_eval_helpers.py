import numpy as np

def hermitian_elements(a, rtol=1e-08, atol=1e-08):
    return np.sum(np.isclose(a, np.conj(a.T), rtol=rtol, atol=atol))

abs_error = lambda h_m, h_n : np.sum(np.abs(h_m - h_n  ))

rel_error = lambda h_m, h_n : np.sum(np.abs(h_m-h_n)) / np.sum(np.abs(h_m))