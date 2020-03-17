import numpy as np
from scipy.special import comb
from itertools import combinations_with_replacement
from functools import * 

COORDINATES = 5
DONALDSON_MAX_ITERATIONS = 10

point_weight_dtype = np.dtype([('point', np.complex64, COORDINATES), ('weight', np.float64)])

def eval_sections(sections, point):
    return np.fromiter(map(lambda monomial: monomial(point), sections), dtype=complex)

def basis_size(k):
    return int(comb(COORDINATES  + k - 1, k) if k <= 0 \
        else (comb(COORDINATES + k - 1, k) - comb(k - 1, k - COORDINATES)))

def generate_quintic_point_weights(k):
    """ (STUB) Generates a structured array of points (on fermat quintic) and associated integration weights """
    n_k = basis_size(k)
    point_weights = np.ones(n_k, dtype=point_weight_dtype)
    sections = monomials(k) 
    return point_weights, sections

def monomials(k):
    """ 
    A set of k degree monomials basis 
    Returns: sections on the complex projector space $P^4$ represented by an array of functions 
    """
    #TODO: need to constrain to hypersurface
    monomial = lambda z, select_indices : np.prod(np.take(z, select_indices))
    for select_indices in combinations_with_replacement(range((COORDINATES - 1) + k - 1), k):
        yield partial(monomial, select_indices=list(select_indices))

def donaldson(k, generator=generate_quintic_point_weights):
    """ Calculates the numerical Calabi-Yau metric on the ambient space $P^4$ """
    point_weights, sections = generate_quintic_point_weights(k)
    n_k = basis_size(k)
    n_p = len(point_weights)
    h_0 = np.zeros((n_k, n_k), dtype=complex)

    volume_cy = lambda pw : (1 / n_p) * np.sum(pw['weight']) # sum weights
    t_operator_func = lambda h_n : (n_k / volume_cy(point_weights)) * t_operator(n_k, h_n, sections, point_weights)

    h_balanced = reduce(lambda h_n, _ : np.transpose(np.linalg.inv(t_operator_func(h_n))), \
        range(1, DONALDSON_MAX_ITERATIONS), h_0)
    return h_balanced

def t_operator(n_k, h_n, sections, point_weights):
    t_acc = np.zeros((n_k, n_k), dtype=complex)
    for p_w in point_weights:
        s_p = eval_sections(sections, p_w['point']) 
        inner = np.einsum('ij,i,j', h_n, s_p, s_p)
        t_acc += np.einsum('i,j', s_p, np.conjugate(s_p))  * p_w['weight'] / inner
    return t_acc

if __name__ == "__main__":
    g = donaldson(k = 2)
    print(g)