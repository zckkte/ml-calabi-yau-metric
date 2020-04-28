import numpy as np
from scipy.special import comb
from scipy import optimize
from itertools import combinations_with_replacement, islice
from functools import * 
import numdifftools as nd

COORDINATES = 5
DONALDSON_MAX_ITERATIONS = 10

point_weight_dtype = np.dtype([
    ('point', np.complex64, COORDINATES), 
    ('weight', np.float64) 
])


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
    return np.concatenate(reduce(lambda acc, _ : acc + [sample_quintic()], 
        range(int(n_p / COORDINATES)), []))

def weight(point):
    z_j = point[find_max_dq_coord_index(point)] 
    w = find_kahler_form(point, z_j)
    return (5 ** -2) * np.abs(z_j) ** (-8) * np.linalg.det(w) ** (-1)

def find_kahler_form(point, z_j):
    jac = np.transpose(jacobian(point, z_j))
    jac_bar = np.conj(jac)
    w_fs_form = fubini_study_kahler_form(point)
    return np.einsum('ia,ij,jb -> ab', jac, w_fs_form, jac_bar)

def jacobian(z, z_j):
    select = (z != z_j) & (affine_coord(z) == False)
    partials = -(z[select] / z_j) ** 4
    return np.array([
        [0, partials[0], 1, 0, 0 ], 
        [0, partials[1], 0, 1, 0 ], 
        [0, partials[2], 0, 0, 1 ]
    ])

def fubini_study_kahler_form(point):
    fubini_study_kahler_pot = lambda p : (1 / np.pi) * np.log(np.sum(np.abs(p) ** 2))
    fs_metric = nd.Hessian(fubini_study_kahler_pot) (point)
    return fs_metric

affine_coord = lambda p : np.isclose(p, np.complex(1, 0)) 

def to_good_coordinates(point):
    """accepts point in affine patch"""
    x = np.copy(point)
    max_dq_index = find_max_dq_coord_index(point) 
    exclude_max_dq_index = (x != x[max_dq_index]) & (affine_coord(x) == False) 
    x[exclude_max_dq_index == False] = 0 
    return x

def find_max_dq_coord_index(point):
    """accepts point in affine patch"""
    dq_abs = lambda p : np.absolute([5 * z ** 4 for z in p])
    dq_abs_max_index = lambda func, p : np.argmax(np.ma.array(func(p),
        mask=np.isclose(p,np.complex(1, 0)) ))
    return dq_abs_max_index(dq_abs, point)

def to_affine_patch(point):
    max_norm_coord = lambda p : p[np.argmax(np.absolute(p))]
    return point / max_norm_coord(point)

def generate_quintic_point_weights(k):
    """ 
    Generates a structured array of points (on fermat quintic in affine coordinates)
        and associated integration weights 
    """
    n_k = basis_size(k)
    n_p = 10 * n_k ** 2 + 50000
    point_weights = np.zeros((n_p), dtype=point_weight_dtype)
    sample_points = sample_quintic_points(n_p)
    weights = np.vectorize(weight, signature="(m)->()")(sample_points)
    point_weights['point'], point_weights['weight'] = sample_points, weights

    return point_weights, monomials(k) 

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
    return np.fromiter(map(lambda monomial: monomial(point), sections), dtype=complex)

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