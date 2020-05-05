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
    jacobian = np.zeros((3,5), dtype=complex)
    select = (z != z_j) & (affine_coord(z) == False)
    partials = -(z[select] / z_j) ** 4
    partial_i = np.where(z == z_j)[0][0]
    diagonal_i = np.where(select)[0]
    for i in range(3): #manifold specific
        jacobian[i][diagonal_i[i]] = 1
        jacobian[i][partial_i] = partials[i]
    return jacobian

def fubini_study_kahler_form(point):
    fubini_study_kahler_pot = lambda p : (1 / np.pi) * np.log(np.sum(np.abs(p) ** 2))
    return nd.Hessian(fubini_study_kahler_pot) (point)

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

def generate_quintic_point_weights(k,  n_t= -1):
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
    return np.fromiter(map(lambda monomial: monomial(point), sections), dtype=complex)

def eval_sections_partial(k, point):
    start_index = int(comb(k - 1, k - COORDINATES)) if k >= COORDINATES else None 
    monomials = np.fromiter(combinations_with_replacement(range(COORDINATES), 2))[start_index::]
    partial_sections = np.zero(COORDINATES, dtype=complex)
    for i in range(COORDINATES): 
        for j in range(basis_size(k)): #monomials
            tmp = 1
            occurrences = 0
            for m in range(k):
                tmp = tmp * point[monomials[j][m]]
                if monomials[j][m] == i:
                    occurrences += 1
            if occurrences != 0:
                partial_sections[i] += tmp * occurrences/ point[i]
    return partial_sections

def initial_balanced_metric(n_k):

    i = 0
    while i < 10:
        h_initial = np.zeros((n_k, n_k), dtype=complex)
        for i in range(n_k - 1):
            for j in range(i + 1, n_k):
                h_initial[i][j] = np.random.rand(1, 2).astype(float).view(np.complex128)
        h_initial += np.conjugate(h_initial.T)

        np.fill_diagonal(h_initial, np.random.rand(n_k))

        h_inverse = np.linalg.inv(h_initial)
        prod = np.einsum('ij,jk', h_initial, h_inverse)
        if np.all(np.isclose(prod, np.eye(n_k, dtype=complex), atol=1e-12)):
            break
        i+=1

    return h_initial

def donaldson(k, generator=generate_quintic_point_weights):
    """ Calculates the numerical Calabi-Yau metric on the ambient space $P^4$ """
    point_weights = generator(k)
    n_k = basis_size(k)
    n_p = len(point_weights)
    h_0 = initial_balanced_metric(n_k)

    volume_cy = lambda pw : (1 / n_p) * np.sum(pw['weight']) # sum weights
    t_operator_func = lambda h_n : (n_k / volume_cy(point_weights)) * t_operator(k, n_k, h_n, point_weights)

    h_balanced = reduce(lambda h_n, _ : np.transpose(np.linalg.inv(t_operator_func(h_n))), \
        range(1, DONALDSON_MAX_ITERATIONS), h_0)
    return h_balanced

def t_operator(k, n_k, h_n, point_weights):
    t_acc = np.zeros((n_k, n_k), dtype=complex)
    for p_w in point_weights:
        s_p = eval_sections(monomials(k), p_w['point']) 
        inner = np.einsum('ij,i,j', h_n, s_p, np.conjugate(s_p))
        t_acc += np.einsum('i,j', s_p, np.conjugate(s_p))  * p_w['weight'] / inner
    return t_acc

def pull_back_metric(k, h_balanced, point):
    z_j = point[find_max_dq_coord_index(point)] 
    jac = jacobian(point, z_j)
    s_p = eval_sections(monomials(k), point) 
    partial_sp = eval_sections_partial(k, point)
    partial_sp_conj = np.conjugate(partial_sp)
    kahler_pot_partial_0 = np.log(np.einsum('ij,i,j', h_balanced, s_p, np.conjugate(s_p)))
    kahler_pot_partial_1 = np.einsum('ab,ia,b', h_balanced, partial_sp, np.conjugate(s_p))
    kahler_pot_partial_2 = np.einsum('ab,ia,jb', h_balanced, partial_sp, partial_sp_conj)
    g_tilde = ((k * np.pi)**(-1) * (kahler_pot_partial_0 * kahler_pot_partial_2 
        - kahler_pot_partial_0 ** 2 * kahler_pot_partial_1 * np.conjugate(kahler_pot_partial_1 )))
    return np.einsum('ai,ij,bj', np.conjugate(jac), g_tilde, jac)

def sigma(k, n_t, g, generator=generate_quintic_point_weights):
    point_weights = generate_quintic_point_weights(k, n_t)
    g_at_point = g(pw)
    volume_cy = (1 / n_t) * np.sum(point_weights['weight']) # sum weights 
    volume_k = (1 / n_t) * np.sum ( np.vectorize(vol_k_integrand(g_at_point))(point_weights) )

    sigma_acc = 0
    for pw in point_weights:
        sigma_acc += np.abs(1 - quintic_kahler_form_determinant(g_at_point(pw), pw) 
            * volume_cy / (omega_wedge_omega_conj(pw['point']) * volume_k ) * pw['weight'])

    return (1 / (n_t * volume_cy)) * sigma_acc

def quintic_kahler_form_determinant(g, pw):
    return np.linalg.det(g) #prefactors?

def omega_wedge_omega_conj(pw):
    point = pw['point']
    z_j = point[find_max_dq_coord_index(point)] 
    return 5 ** (-2) * np.abs(z_j) ** (-8)

def vol_k_integrand(g, pw):
    omega3 = quintic_kahler_form_determinant (g, pw['point'])
    omega_squared = omega_wedge_omega_conj(pw['point'])
    return omega3 / omega_squared  * pw['weight']

if __name__ == "__main__":
    k = 2
    n_t = 500000
    h_balanced = donaldson(k)
    g = lambda pw : pull_back_metric(k, h_balanced, pw)
    measure = sigma(k, n_t, g)
    print(h_balanced)
    print(measure)