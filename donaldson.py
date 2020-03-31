import numpy as np
from scipy.special import comb
from scipy import optimize
from itertools import combinations_with_replacement, islice
from functools import * 

COORDINATES = 5
DONALDSON_MAX_ITERATIONS = 10

point_weight_dtype = np.dtype([
    ('point', np.complex64, COORDINATES), 
    ('weight', np.float64) 
])

def eval_sections(sections, point):
    return np.fromiter(map(lambda monomial: monomial(point), sections), dtype=complex)

def basis_size(k):
    return int(comb(COORDINATES + k - 1, k) if k < COORDINATES \
        else (comb(COORDINATES + k - 1, k) - comb(k - 1, k - COORDINATES)))

def compose(*functions):
    return reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)

def sample_ambient_pair():
    """two distinct random points in ambient $P^4$ space"""
    n = 9
    p, q = np.split(np.random.normal(0, 1, 2 * (n + 1)), 2)
    normalise = lambda r : r / (np.sum(r ** 2) ** (0.5))
    to_complex_proj = (lambda v : v.reshape((5, 2)).astype(float)
        .view(np.complex128)
        .reshape(5))
    to_ambient = compose(to_complex_proj, normalise)
    return to_ambient(p), to_ambient(q)

def sample_quintic():
    """samples 5 points from fermat quintic """
    p, q = sample_ambient_pair()
    line = lambda t, p, q : (p + q * t) 
    quintic_points = lambda ts, p, q : np.array([ line(t, p, q) for t in ts ]) 
    quintic_intersect = lambda t, *points : np.sum( line(t, points[0], points[1]) ** 5)  
    sol = optimize.fsolve(quintic_intersect, 0., args=(p, q))
    #quintic_inter_solve = lambda part, initial : optimize.fsolve(part(quintic_intersect), initial, args=(p, q))
    #quintic_inter_solve(np.real, 0.), quintic_inter_solve(np.imag, 0.)
    return quintic_points(sol, p, q)

def generate_quintic_point_weights(k):
    """ (STUB) Generates a structured array of points (on fermat quintic) and associated integration weights """
    n_k = basis_size(k)
    point_weights = np.ones(n_k, dtype=point_weight_dtype)
    sections = monomials(k) 
    return point_weights, sections

def monomials(k):
    """ 
    A set of k degree monomials basis 
    Returns: sections (represented by a partial function generator) 
        on the complex projection space $P^4$ constrained to the fermat quintic 
    """
    start_index = int(comb(k - 1, k - COORDINATES)) if k >= COORDINATES else None 
    monomial_index_iter = islice(combinations_with_replacement(range(COORDINATES), k), start_index, None)
    for select_indices in monomial_index_iter:
        yield partial(lambda z, select_indices : np.prod(np.take(z, select_indices)), 
            select_indices=list(select_indices)) 

def donaldson(k, generator=generate_quintic_point_weights):
    """ Calculates the numerical Calabi-Yau metric on the ambient space $P^4$ """
    point_weights, sections = generator(k)
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