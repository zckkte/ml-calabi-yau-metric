import donaldson as don
import numpy as np
from functools import * 

def sigma(k, n_t, g_pull_back, generator=don.generate_quintic_point_weights):
    point_weights = generator(k, n_t)
    vol_cy = volume_cy(n_t, point_weights)
    vol_k = volume_k(n_t, point_weights, g_pull_back)

    return (n_t * vol_cy) ** (-1) * reduce(lambda acc, pw : 
        acc + np.abs(1 - quintic_kahler_form_determinant(g_pull_back(pw['point'])) 
            * vol_cy / (omega_wedge_omega_conj(pw['point']) * vol_k ) * pw['weight']), 
        point_weights, 0)

def global_ricci_scalar (k, n_t, g_pull_back, generator=don.generate_quintic_point_weights):
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

volume_cy = lambda n_t, point_weights : (1 / n_t) * np.sum(point_weights['weight']) # sum weights 

volume_k = (lambda n_t, point_weights, g_pull_back : 
    (1 / n_t) * np.sum ( np.vectorize(vol_k_integrand, signature='(),()->()')(point_weights, g_pull_back) ))

def vol_k_integrand(point_weight, g_pull_back):
    point, weight = point_weight
    omega3 = quintic_kahler_form_determinant (g_pull_back(point))
    omega_squared = omega_wedge_omega_conj(point)
    return (omega3 / omega_squared) * weight

quintic_kahler_form_determinant = lambda g : np.linalg.det(g) #prefactors?

omega_wedge_omega_conj = lambda point : 5 ** (-2) * np.abs(don.elim_z_j(point)) ** (-8)
