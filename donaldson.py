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
    w = find_kahler_form(point)
    z_j = elim_z_j(point)
    return (5 ** -2) * np.abs(z_j) ** (-8) * np.linalg.det(w) ** (-1)

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
    fubini_study_kahler_pot = lambda p : (1 / np.pi) * np.log(np.sum(np.abs(p) ** 2))
    return nd.Hessian(fubini_study_kahler_pot) (point)

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
    weights = np.vectorize(weight, signature="(m)->()")(sample_points)

    point_weights = np.zeros((n_p), dtype=point_weight_dtype)
    point_weights['point'], point_weights['weight'] = sample_points, weights

    return point_weights

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

def monomial_partials(k):
    for section in monomials(k):
        yield nd.Jacobian(lambda p : section(p))

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
    return np.all(np.isclose(prod, np.eye(dim, dtype=complex), atol=1e-12))

def initial_balanced_metric(n_k):
    for _ in range(10):
        h_initial = triu_exclude_diag((n_k, n_k),  
            value=np.random.rand(1, 2).astype(float).view(np.complex128),
            dtype=complex)
        h_initial += np.conjugate(h_initial.T)
        np.fill_diagonal(h_initial, np.random.rand(n_k))

        if is_invertible(h_initial):
            break
    return h_initial

def donaldson(k, max_iterations=10, generator=generate_quintic_point_weights):
    """ Calculates the numerical Calabi-Yau metric on the ambient space $P^4$ """
    point_weights = generator(k)
    n_k = basis_size(k)
    n_p = len(point_weights)
    h_0 = initial_balanced_metric(n_k)

    volume_cy = lambda pw : (1 / n_p) * np.sum(pw['weight']) # sum weights
    t_operator_func = lambda h_n : (n_k / (n_p * volume_cy(point_weights))) * t_operator(k, n_k, h_n, point_weights)

    h_balanced = reduce(lambda h_n, _ : np.transpose(np.linalg.inv(t_operator_func(h_n))), \
        range(1, max_iterations), h_0)
    return h_balanced

def t_operator(k, n_k, h_n, point_weights):
    t_acc = np.zeros((n_k, n_k), dtype=complex)
    for p_w in point_weights:
        s_p = eval_sections(monomials(k), p_w['point']) 
        inner = np.einsum('ij,i,j', h_n, s_p, np.conjugate(s_p))
        t_acc += np.einsum('i,j', s_p, np.conjugate(s_p))  * p_w['weight'] / inner
    return t_acc

def pull_back_metric(k, h_balanced, point):
    s_p = eval_sections(monomials(k), point) 
    partial_sp = eval_sections(monomial_partials(k), point)
    partial_sp_conj = np.conjugate(partial_sp)
    kahler_pot_partial_0 = np.log(np.einsum('ij,i,j', h_balanced, s_p, np.conjugate(s_p)))
    kahler_pot_partial_1 = np.einsum('ab,ai,b', h_balanced, partial_sp, np.conjugate(s_p))
    kahler_pot_partial_2 = np.einsum('ab,ai,bj', h_balanced, partial_sp, partial_sp_conj)
    g_tilde = ((k * np.pi)**(-1) * (kahler_pot_partial_0 * kahler_pot_partial_2 
        - (kahler_pot_partial_0 ** 2) * kahler_pot_partial_1 * np.conjugate(kahler_pot_partial_1 )))
    jac = jacobian(point)
    return np.einsum('ai,ij,bj', np.conjugate(jac), g_tilde, jac)

def sigma(k, n_t, g_pull_back, generator=generate_quintic_point_weights):
    point_weights = generator(k, n_t)
    vol_cy = volume_cy(n_t, point_weights)
    vol_k = volume_k(n_t, point_weights, g_pull_back)

    return (n_t * vol_cy) ** (-1) * reduce(lambda acc, pw : 
        acc + np.abs(1 - quintic_kahler_form_determinant(g_pull_back(pw['point'])) 
            * vol_cy / (omega_wedge_omega_conj(pw['point']) * vol_k ) * pw['weight']), 
        point_weights, 0)

volume_cy = lambda n_t, point_weights : (1 / n_t) * np.sum(point_weights['weight']) # sum weights 

volume_k = (lambda n_t, point_weights, g_pull_back : 
    (1 / n_t) * np.sum ( np.vectorize(vol_k_integrand, signature='(),()->()')(point_weights, g_pull_back) ))

quintic_kahler_form_determinant = lambda g : np.linalg.det(g) #prefactors?

omega_wedge_omega_conj = lambda point : 5 ** (-2) * np.abs(elim_z_j(point)) ** (-8)

def vol_k_integrand(point_weight, g_pull_back):
    point, weight = point_weight
    omega3 = quintic_kahler_form_determinant (g_pull_back(point))
    omega_squared = omega_wedge_omega_conj(point)
    return (omega3 / omega_squared) * weight

def global_ricci_scalar (k, n_t, g_pull_back, generator=generate_quintic_point_weights):
    point_weights = generator(k, n_t)
    vol_cy = volume_cy(n_t, point_weights)
    vol_k_3 = volume_k(n_t, point_weights, g_pull_back) ** (1/3)

    return (vol_k_3 / (n_t * vol_cy)) * reduce(lambda pw, acc :
        acc + quintic_kahler_form_determinant(g_pull_back(pw['point'])) 
            / omega_wedge_omega_conj(pw['point']) * ricci_scalar_k(pw['point'], g_pull_back) * pw['weight'], 
        point_weights, 0.)

def ricci_scalar_k(point, g_pull_back):
    """STUB"""
    return 0.

"""helpers"""

read_point_weights_from_file = lambda file_name : (lambda _ : np.fromfile(file_name, dtype=point_weight_dtype))

def save_generate_point_weights(k, file_name='sample_data'):
    np.save("%s_%d" % (file_name, k), generate_quintic_point_weights(k))

def load_h_balanced(file_name, k):
    dim = basis_size(k)
    return (np.fromfile().reshape((dim * dim, k))
        .view(np.complex128).reshape(dim, dim))

if __name__ == "__main__":
    k = 2
    n_t = 500000
    #h_balanced = donaldson(k, generator=read_point_weights_from_file('pw_2_15.dat'))
    h_balanced = load_h_balanced('h_balanced_k_2_15.dat', k)
    measure = sigma(k, n_t, lambda pw : pull_back_metric(k, h_balanced, pw), \
        generator= lambda k, n_t : read_point_weights_from_file('pw_2_15.dat')(k)[:n_t])
    print(h_balanced) 
    print('sigma=', measure) 