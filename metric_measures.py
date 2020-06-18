import donaldson as don
import fermat_quintic as fq
import numpy as np
import numdifftools as nd
from joblib import Parallel, delayed
from functools import * 

def sigma(k, n_t, h_balanced, generator=fq.generate_quintic_point_weights):
    point_weights = generator(k, n_t)
    g_pull_back = lambda p : fq.pull_back(k, h_balanced, p)
    vol_cy = volume_cy(n_t, point_weights)
    vol_k = volume_k(n_t, point_weights, g_pull_back)
        
    sigma_integrand = np.vectorize(lambda pw : np.abs(1 - quintic_kahler_form_determinant(g_pull_back(pw['point'])) 
            * vol_cy / (omega_wedge_omega_conj(pw['point']) * vol_k ) * pw['weight']), 
                signature='()->()')
    with Parallel(n_jobs=-1, prefer='processes') as parallel:
        t_acc_part = parallel(delayed(sigma_integrand) (point_weight) for point_weight in point_weights)
        return (n_t * vol_cy) ** (-1) * sum(t_acc_part)

def global_ricci_scalar (k, n_t, h_balanced, generator=fq.generate_quintic_point_weights):
    point_weights = generator(k, n_t)
    g_pull_back = lambda p : fq.pull_back(k, h_balanced, p)
    vol_cy = volume_cy(n_t, point_weights)
    vol_k_3 = volume_k(n_t, point_weights, g_pull_back) ** (1/3)

    return (vol_k_3 / (n_t * vol_cy)) * reduce(lambda acc, pw :
        acc + quintic_kahler_form_determinant(g_pull_back(pw['point'])) 
            / omega_wedge_omega_conj(pw['point']) * ricci_scalar_k(k, h_balanced, g_pull_back, pw['point']) * pw['weight'], 
        point_weights, 0.)

def ricci_scalar_k(k, h_balanced, g_pull_back, point):
    g_kahler = fq.kahler_metric(k, h_balanced, point)
    partial_g_k = compute_kahler_metric_partial(k, h_balanced, point)
    double_partial_g_k = compute_kahler_metric_double_partial(k, h_balanced, point)
    jac = np.transpose(fq.jacobian(point))
    jac_bar = np.conj(jac)
    partial_jac = nd.Gradient(lambda x : fq.jacobian(x))(point)
    partial_jac_conj = np.conj(partial_jac)

    partial_g_pb = (np.einsum('kai,ij,bj->kab', partial_jac, g_kahler, jac_bar)
        + np.einsum('ai,kij,bj->kab', jac, partial_g_k, jac_bar))
    double_partial_g_pb = (np.einsum('kai,ij,hbj', partial_jac, g_kahler, partial_jac_conj) 
        + np.einsum('mai,mij,bj', partial_jac, partial_g_k, jac_bar) 
        + np.einsum('ai,ijnm,bj', jac, double_partial_g_k, jac_bar) 
        + np.einsum('ai,nij,mbj', jac, partial_g_k, partial_jac_conj))

    g_pb_inv = np.inverse(g_pull_back(point))
    ricci = np.trace( (-1) * g_pb_inv * partial_g_pb * g_pb_inv * partial_g_pb  
        + g_pb_inv * double_partial_g_pb)
    return np.trace(ricci)

def compute_kahler_metric_partial(k, h_balanced, point):
    """STUB"""
    s_p = fq.eval_sections(fq.monomials(k), point) 
    partial_sp = fq.eval_with(lambda s: nd.Jacobian(s)(point), fq.monomials(k)) 
    partial_sp_conj = np.conjugate(partial_sp)
    
    return np.ones((3,5,5))

def compute_kahler_metric_double_partial(k, h_balanced, point):
    """STUB"""
    return np.ones((3,5,5))

volume_cy = lambda n_t, point_weights : (1 / n_t) * np.sum(point_weights['weight']) # sum weights 

volume_k = (lambda n_t, point_weights, g_pull_back : 
    (1 / n_t) * np.sum ( np.vectorize(vol_k_integrand, signature='(),()->()')(point_weights, g_pull_back) ))

def vol_k_integrand(point_weight, g_pull_back):
    point, weight = point_weight
    omega3 = quintic_kahler_form_determinant (g_pull_back(point))
    omega_squared = omega_wedge_omega_conj(point)
    return (omega3 / omega_squared) * weight

quintic_kahler_form_determinant = lambda g : np.linalg.det(g) #prefactors?

omega_wedge_omega_conj = lambda point : 5 ** (-2) * np.abs(fq.elim_z_j(point)) ** (-8)
