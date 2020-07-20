from joblib import Parallel, delayed
from functools import * 
import tensorflow as tf

COORDINATES = 5

"""
    g_point_weights = [ [...point, ...weight, ...determinant ], ...]
"""

def sigma_error(g_point_weights : tf.Tensor) -> tf.float64:
    """ tensorflow implementation of sigma error """
    n_t = len(g_point_weights)
    weights = tf.math.real(g_point_weights[:, COORDINATES])
    vol_cy = volume_cy(weights)
    vol_k = volume_k(n_t, g_point_weights)

    return ((n_t * vol_cy) ** (-1) * tf.foldl(lambda acc, gpw : 
            acc + tf.abs(1 - __determinant(gpw) * vol_cy / (omega_wedge_omega_conj(__point(gpw)) * vol_k )) * __weight(gpw), 
            g_point_weights, 0.))

volume_k = lambda n_t, g_point_weights : (1 / n_t) * tf.foldl(lambda acc, gpw : acc + vol_k_integrand(gpw), g_point_weights, 0.) 

elim_z_j = lambda z : (-1) - tf.reduce_sum(z[good_coord_mask(z)] ** 5)

good_coord_mask = lambda x: (x != x[find_max_dq_coord_index(x)]) & exclude_affine_mask(x) 

find_max_dq_coord_index = lambda p : tf.argmax(tf.abs(tf.math.pow(tf.boolean_mask(p, exclude_affine_mask(p)), 4)))

exclude_affine_mask = lambda p : (tf.math.equal(p, tf.cast(tf.complex(1.,0.), dtype=tf.complex128)) == False)

omega_wedge_omega_conj = lambda point : 5 ** (-2) * tf.abs(elim_z_j(point)) ** (-8)

volume_cy = lambda weights : (1 / len(weights)) * tf.math.reduce_sum(weights) 

__point = lambda g_point_weight : g_point_weight[0:COORDINATES] 

__weight = lambda g_point_weight : tf.math.real(g_point_weight[COORDINATES] )

__determinant = lambda g_point_weight : tf.math.real(g_point_weight[-1])

def vol_k_integrand(g_point_weight):
    p, w, det = (__point(g_point_weight), __weight(g_point_weight), __determinant(g_point_weight))
    return tf.math.divide(det, omega_wedge_omega_conj(p)) * w
