import donaldson as don
import fermat_quintic as fq
import numpy as np
import numdifftools as nd
from joblib import Parallel, delayed
from functools import * 

metric_point_weight_dtype = np.dtype([
    ('metric', np.complex64, (3, 3)), 
    ('point_weight', fq.point_weight_dtype), 
])

def sigma_error(g_point_weights):
    """ Calculates sigma error for a given set of 3-tuples containing point-weight pairs and corresponding local pull-back metric """
    n_t = len(g_point_weights)
    vol_cy = volume_cy(g_point_weights['point_weight'])
    #dummy lambda necessary when calculating `g_pull_back` prior to calling `sigma_error`
    vol_k = (1 / n_t) * np.sum([ vol_k_integrand(gpw['point_weight'], (lambda _ : gpw['metric'])) 
        for gpw in g_point_weights ]) 
    sigma_integrand = np.vectorize(lambda gpw : np.abs(1 - quintic_kahler_form_determinant(gpw['metric']) 
            * vol_cy / (omega_wedge_omega_conj(gpw['point_weight']['point']) * vol_k ) ) * gpw['point_weight']['weight'], 
            signature='()->()')
    return (n_t * vol_cy) ** (-1) * sum(sigma_integrand(g_point_weights))

def __sigma_error(g_pull_back, point_weights):
    vol_cy = volume_cy(point_weights)
    vol_k = volume_k(point_weights, g_pull_back)
    n_t = len(point_weights)

    sigma_integrand = np.vectorize(lambda pw : np.abs(1 - quintic_kahler_form_determinant(g_pull_back(pw['point'])) 
            * vol_cy / (omega_wedge_omega_conj(pw['point']) * vol_k )) * pw['weight'], 
                signature='()->()')
    with Parallel(n_jobs=-1, prefer='processes') as parallel:
        sigma_acc_part = parallel(delayed(sigma_integrand) (point_weight) for point_weight in point_weights)
        return (n_t * vol_cy) ** (-1) * sum(sigma_acc_part)

def sigma(k, n_t, h_balanced, generator=fq.generate_quintic_point_weights):
    return __sigma_error(g_pull_back=lambda p : fq.pull_back(k, h_balanced, p), 
        point_weights=generator(k, n_t))

def global_ricci_scalar (k, n_t, h_balanced, generator=fq.generate_quintic_point_weights):
    point_weights = generator(k, n_t)
    g_pull_back = lambda p : fq.pull_back(k, h_balanced, p)
    vol_cy = volume_cy(point_weights)
    vol_k_3 = volume_k(point_weights, g_pull_back) ** (1/3)
            
    ricci_integrand = np.vectorize(lambda pw : quintic_kahler_form_determinant(g_pull_back(pw['point'])) 
            / omega_wedge_omega_conj(pw['point']) * np.abs(ricci_scalar_k(k, h_balanced, g_pull_back, pw['point'])) * pw['weight'], 
                signature='()->()')
    with Parallel(n_jobs=-1, prefer='processes') as parallel:
        ricci_int_acc_part = parallel(delayed(ricci_integrand) (point_weight) for point_weight in point_weights)
        return (vol_k_3 / (n_t * vol_cy)) * sum(ricci_int_acc_part)

def ricci_scalar_k(k, h_balanced, g_pull_back, point):
    g_kahler = fq.kahler_metric(k, h_balanced, point)
    k_pot = kahler_pot_partials(k, h_balanced, point)
    partial_g_k = compute_kahler_metric_partial(k, h_balanced, k_pot, point)
    double_partial_g_k = compute_kahler_metric_double_partial(k, h_balanced, k_pot, point)
    jac = np.transpose(fq.jacobian(point))
    jac_bar = np.conj(jac)
    partial_jac = nd.Gradient(lambda x : fq.jacobian(x))(point)
    partial_jac_conj = np.conj(partial_jac)

    partial_g_pb =  (np.einsum('aki,ij,jb', partial_jac, g_kahler, jac_bar)
        + np.einsum('ia,kij,jb', jac, partial_g_k, jac_bar))
    double_partial_g_pb = (np.einsum('ami,nij,jb', partial_jac, partial_g_k, jac_bar) 
        + np.einsum('ia,ijnm,jb', jac, double_partial_g_k, jac_bar) 
        + np.einsum('kai,ij,hbj->khab', partial_jac, g_kahler, partial_jac_conj) 
        + np.einsum('ia,nij,bmj', jac, partial_g_k, partial_jac_conj))

    g_pb_inv = np.linalg.inv(g_pull_back(point))
    ricci = (k * np.pi) ** (-1) * np.trace((-1) * np.einsum('ab,bci,cd,dej', g_pb_inv, partial_g_pb, g_pb_inv, partial_g_pb) 
        + np.einsum('ab,bcij', g_pb_inv, double_partial_g_pb))
    return np.trace(np.einsum('ia,jb,ij', jac, jac_bar, ricci))

def kahler_pot_partials (k, h_bal, point) : 
    s_p = fq.eval_sections(fq.monomials(k), point) 
    partial_sp = fq.eval_with(lambda s: nd.Jacobian(s)(point), fq.monomials(k)) 
    double_partial_sp = fq.eval_with(lambda s: nd.Hessian(s)(point), fq.monomials(k)) 
    k_0 = np.real(fq.kahler_pot_0(h_bal, s_p))
    k_1 = fq.kahler_pot_partial_1(h_bal, partial_sp, s_p)
    k_2 = np.einsum('ab,aij,b', h_bal, double_partial_sp, np.conj(s_p))
    k_3 = np.einsum('ab,aij,bk', h_bal, double_partial_sp, np.conj(partial_sp))
    k_4 = np.einsum('ab,aik,bjl', h_bal, double_partial_sp, np.conj(double_partial_sp))
    return [ k_0, k_1, k_2, k_3, k_4 ]

def compute_kahler_metric_partial(k, h_bal, k_pot, point):
    """(B.78) Ashmore"""
    return ((-1) * ( k_pot[0] ** 2 ) * 
            (np.einsum('i,kl', k_pot[1], k_pot[2]) + np.einsum('k,il->ikl', k_pot[1], k_pot[2]) 
                + np.einsum('l,ik->ikl', np.conj(k_pot[1]), k_pot[2]) ) 
            + k_pot[0] * k_pot[3] + 2 * k_pot[0] ** 3 * np.einsum('i,k,l->ikl', k_pot[1], k_pot[1], np.conj(k_pot[1])))

def compute_kahler_metric_double_partial(k, h_bal, k_pot, point):
    """(B.81) Ashmore"""
    return (k_pot[0] * k_pot[4] 
        - (k_pot[0] ** 2) * (np.einsum('ij,kl', k_pot[2], k_pot[2]) 
            + np.einsum('ij,kl', k_pot[2], np.conj(k_pot[2])) + np.einsum('ij,kl', k_pot[2], k_pot[2]))
        - (k_pot[0] ** 2) * (np.einsum('j, ikl', np.conj(k_pot[1]), k_pot[3]) 
            + np.einsum('j, ikl', np.conj(k_pot[1]), k_pot[3]) 
            + np.einsum('j, ikl', k_pot[1], np.conj(k_pot[3]))
            + np.einsum('j, ikl', k_pot[1], np.conj(k_pot[3])))
        + 2 * k_pot[0] ** 3 * (np.einsum('i,j,kl', k_pot[1], np.conj(k_pot[1]), k_pot[2]) 
            + np.einsum('ij,k,l', k_pot[2], k_pot[1], np.conj(k_pot[1])) 
            + np.einsum('j,k,il', np.conj(k_pot[1]), k_pot[1], k_pot[2]) 
            + np.einsum('i,kj,l', k_pot[1], k_pot[2], np.conj(k_pot[1])) 
            + np.einsum('i,k,jl', k_pot[1], k_pot[1], np.conj(k_pot[2]))
            + np.einsum('j,ik,l', np.conj(k_pot[1]), k_pot[2], np.conj(k_pot[1])))
        - 6 * k_pot[0] ** 4 * np.einsum('i,j,k,l', k_pot[1], np.conj(k_pot[1]), k_pot[1],np.conj(k_pot[1])))

volume_cy = lambda point_weights : (1 / len(point_weights)) * np.sum(point_weights['weight']) 

volume_k = (lambda point_weights, g_pull_back : 
    (1 / len(point_weights)) * np.sum(np.vectorize(vol_k_integrand, signature='(),()->()')(point_weights, g_pull_back)))

def vol_k_integrand(point_weight, g_pull_back):
    point, weight = point_weight
    omega3 = quintic_kahler_form_determinant (g_pull_back(point))
    omega_squared = omega_wedge_omega_conj(point)
    return (omega3 / omega_squared) * weight

quintic_kahler_form_determinant = lambda g : np.linalg.det(g) #prefactors?

omega_wedge_omega_conj = lambda point : 5 ** (-2) * np.abs(fq.elim_z_j(point)) ** (-8)
