import numdifftools as nd
import numpy as np
from functools import * 
from itertools import combinations_with_replacement, islice
from joblib import Parallel, delayed
from scipy.special import comb

COORDINATES = 5
n_jobs = -2

point_weight_dtype = np.dtype([
    ('point', np.complex64, COORDINATES), 
    ('weight', np.float64) 
])

def basis_size(k):
    return int(comb(COORDINATES + k - 1, k) if k < COORDINATES \
        else (comb(COORDINATES + k - 1, k) - comb(k - 1, k - COORDINATES)))

def compose(*functions):
    return reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)

def weight(point):
    w = find_kahler_form(point)
    z_j = elim_z_j(point)
    return np.real( (5 ** -2) * (np.abs(z_j) ** (-8)) * (np.linalg.det(w) ** (-1)) )

def find_kahler_form(point):
    jac = np.transpose(jacobian(point))
    jac_bar = np.conj(jac)
    w_fs_form = fubini_study_kahler_form(point)
    return np.einsum('ia,ij,jb -> ab', jac, w_fs_form, jac_bar)

elim_z_j = lambda z : (-1) - np.sum(z[good_coord_mask(z)] ** 5)

def jacobian(z):
    select = good_coord_mask(z)
    partials = -(z[select] / elim_z_j(z)) ** 4
    partial_i = np.where(z == z[find_max_dq_coord_index(z)])[0][0]
    diagonal_i = np.where(select)[0]

    jacobian = np.zeros((COORDINATES-2,COORDINATES), dtype=complex)
    for i in range(COORDINATES-2): #manifold specific
        jacobian[i][diagonal_i[i]] = 1
        jacobian[i][partial_i] = partials[i]
    return jacobian

def fubini_study_kahler_form(point):
    return ((1 / np.pi) * (np.sum(np.abs(point) ** 2) ) ** (-2) 
        * ( (np.sum(np.abs(point) ** 2)) * np.eye(COORDINATES) - np.outer(np.conj(point), point) ))

affine_coord = lambda p : np.isclose(p, np.complex(1, 0)) 

good_coord_mask = lambda x: (x != x[find_max_dq_coord_index(x)]) & (affine_coord(x) == False) 

max_dq_coord = lambda p : p[find_max_dq_coord_index(p)]  

def find_max_dq_coord_index(point):
    """accepts point in affine patch"""
    dq_abs = lambda p : np.absolute([5 * z ** 4 for z in p])
    dq_abs_max_index = lambda func, p : np.argmax(np.ma.array(func(p),
        mask=np.isclose(p,np.complex(1, 0)) ))
    return dq_abs_max_index(dq_abs, point)

def to_affine_patch(point):
    max_norm_coord = lambda p : p[np.argmax(np.absolute(p))]
    return point / max_norm_coord(point)

def generate_quintic_point_weights(k, n_t=-1):
    """ 
    Generates a structured array of points (on fermat quintic in affine coordinates)
        and associated integration weights 
    """
    n_k = basis_size(k)
    n_p =  10 * n_k ** 2 + 50000 if n_t < 0 else n_t
    sample_points = sample_quintic_points(n_p)
    weights = Parallel(n_jobs, prefer="processes")(delayed(weight)(p) 
            for p in sample_points)

    point_weights = np.zeros((n_p), dtype=point_weight_dtype)
    point_weights['point'], point_weights['weight'] = sample_points, weights
    return point_weights

def sample_ambient_pair():
    """two distinct random points in ambient $P^4$ space"""
    n = 9
    p, q = np.split(np.random.normal(0, 1, 2 * (n + 1)), 2)
    normalise = lambda r : r / (np.sum(r ** 2) ** (0.5))
    # map to P^4
    to_complex_proj = (lambda v : v.reshape((5, 2)).astype(float)
        .view(np.complex128)
        .reshape(5))
    to_ambient = compose(to_complex_proj, normalise)
    return to_ambient(p), to_ambient(q)

def sample_quintic():
    """samples 5 points from fermat quintic in affine coordinates"""
    p, q = sample_ambient_pair()
    quintic_intersect_coeff = lambda p, q : [ np.sum(comb(COORDINATES, i) * p ** (COORDINATES - i) * q ** i) 
        for i in range(COORDINATES + 1) ] 
    roots = np.roots(quintic_intersect_coeff(q, p))
    return [ to_affine_patch(p + q * t) for t in roots ]

def sample_quintic_points(n_p):
    with Parallel(n_jobs) as parallel:
        return np.concatenate(parallel(delayed(sample_quintic)() 
            for _ in range(int(n_p / COORDINATES))))

def monomials(k):
    """ 
    A set of k degree monomials basis 
    Returns: sections (represented by a partial function generator) 
        on the complex projection space $P^4$ constrained to the fermat quintic 
    """
    start_index = int(comb(k - 1, k - COORDINATES)) if k >= COORDINATES else None 
    monomial_index_iter = islice(combinations_with_replacement(range(COORDINATES), k), 
        start_index, None)
    for select_indices in monomial_index_iter:
        yield partial(lambda z, select_indices : np.prod(np.take(z, select_indices)), 
            select_indices=list(select_indices)) 

def eval_sections(sections, point):
    return np.array(list(map(lambda monomial : np.squeeze(monomial(point)), sections)))

def eval_sections_parallel(sections, point):
    with Parallel(n_jobs) as parallel:
        return np.array(parallel(delayed(lambda monomial : np.squeeze(monomial(point)))(s) for s in sections))

def monomial_partials(k):
    for section in monomials(k):
        yield nd.Jacobian(lambda p : section(p))

def pull_back(k, h_balanced, point):
    jac = jacobian(point)
    g_k = kahler_metric(k, h_balanced, point)
    return np.einsum('ai,ij,bj', jac, g_k, np.conjugate(jac))

kahler_pot_partial_0 = lambda h_bal, s_p : np.log(np.einsum('ij,i,j', h_bal, s_p, np.conjugate(s_p)))

kahler_pot_partial_1 = lambda h_bal, partial_sp, s_p : np.einsum('ab,ai,b', h_bal, partial_sp, np.conjugate(s_p))

kahler_pot_partial_1_bar = lambda h_bal, partial_sp, s_p : np.einsum('ab,a,bi', h_bal, s_p, np.conjugate(partial_sp) )

kahler_pot_partial_2 = lambda h_bal, partial_sp : np.einsum('ab,ai,bj', h_bal, partial_sp, np.conjugate(partial_sp))

def kahler_metric (k, h_bal, point): 
    s_p = eval_sections(monomials(k), point) 
    partial_sp = eval_sections(monomial_partials(k), point)
    k_0 = kahler_pot_partial_0 (h_bal, s_p)
    k_1 = kahler_pot_partial_1 (h_bal, partial_sp, s_p)
    k_1_bar = kahler_pot_partial_1_bar (h_bal, partial_sp, s_p)
    k_2 = kahler_pot_partial_2 (h_bal, partial_sp)
    return (k * np.pi) ** (-1) * (k_0 * k_2 - (k_0 ** 2) * np.einsum('i,j', k_1, k_1_bar))

pull_back_determinant = lambda k, h_balanced, point : np.linalg.det(pull_back(k, h_balanced, point))
