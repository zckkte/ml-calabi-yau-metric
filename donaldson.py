import argparse
import numdifftools as nd
import numpy as np
from functools import * 
from itertools import combinations_with_replacement, islice
from joblib import Parallel, delayed
from scipy.special import comb
import fermat_quintic 

DONALDSON_MAX_ITERATIONS = 10
PROCESSES = -2

def triu_exclude_diag(shape, value, dtype=int):
    n_k, _ = shape
    arr = np.zeros((n_k, n_k), dtype=dtype)
    for i in range(n_k - 1):
        for j in range(i + 1, n_k):
            arr[i][j] = value
    return arr

def is_invertible(arr):
    dim, _ = arr.shape
    arr_inverse = np.linalg.inv(arr)
    prod = np.einsum('ij,jk', arr, arr_inverse)
    return np.all(np.isclose(prod, np.eye(dim, dtype=np.complex), atol=1e-12))

def initial_balanced_metric(n_k):
    for _ in range(10):
        h_initial = triu_exclude_diag((n_k, n_k),  
            value=np.random.rand(1, 2).astype(float).view(np.complex128),
            dtype=complex)
        h_initial += np.conjugate(h_initial.T)
        np.fill_diagonal(h_initial, np.random.rand(n_k))

        if is_invertible(h_initial):
            break
    return h_initial / np.linalg.norm(h_initial)

def donaldson(k, max_iterations=10, generator=fermat_quintic.generate_quintic_point_weights):
    """ Calculates the numerical Calabi-Yau metric on the ambient space $P^4$ """
    point_weights = generator(k)
    n_k = fermat_quintic.basis_size(k)
    n_p = len(point_weights)

    volume_cy = (1 / n_p) * np.sum(point_weights['weight']) 
    t_operator_func = lambda h_new : (n_k / (n_p * volume_cy)) * t_operator(k, n_k, h_new, point_weights)
    return reduce (lambda h_new, _ : np.linalg.inv(t_operator_func(h_new)).T, 
        range(0, max_iterations),
        initial_balanced_metric(n_k))

def t_operator(k, n_k, h_n, point_weights):
    with Parallel(n_jobs=PROCESSES, verbose=True, prefer='processes') as parallel:
        t_acc = np.zeros((n_k, n_k), dtype=np.complex64)
        res = parallel(delayed(t_operator_integrand) (k, h_n, point_weight) 
            for point_weight in point_weights)
        t_acc += sum(res)
        return t_acc

def t_operator_integrand(k, h_n, point_weight):
    s_p = fermat_quintic.eval_sections(fermat_quintic.monomials(k), point_weight['point']) 
    inner = np.einsum('ij,i,j', h_n, s_p, np.conjugate(s_p))
    return np.einsum('i,j', s_p, np.conjugate(s_p))  * point_weight['weight'] / np.real(inner)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Numerically approximate the Calabi-Yau metric of the fermat quintic')
    parser.add_argument('-k', type=int,required=True, help='order of fermat quintic sections')
    args = parser.parse_args()

    sample_point = fermat_quintic.sample_quintic()[0]
    h_bal = donaldson(args.k, max_iterations=12)
    g_pb = fermat_quintic.pull_back(args.k, h_bal, sample_point)
    print('h_bal^(%d)=%f' % (args.k, np.linalg.det(h_bal)))
    print('g^(%d)|_p=%f' % (args.k , np.linalg.det(g_pb)))
