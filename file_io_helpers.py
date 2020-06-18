import numpy as np
import fermat_quintic as fq

read_point_weights_from_file = lambda file_name : (lambda _ : np.fromfile(file_name, dtype=fq.point_weight_dtype))

def save_generate_point_weights(k, file_name='sample_data'):
    np.save("%s_%d" % (file_name, k), fq.generate_quintic_point_weights(k))

def load_balanced_metric(file_name, k):
    dim = fq.basis_size(k)
    return (np.fromfile(file_name).reshape((dim * dim, k))
        .view(np.complex128).reshape(dim, dim))
