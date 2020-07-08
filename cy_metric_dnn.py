import numpy as np
import keras
from keras import layers
import keras.backend as kb
import fermat_quintic as fq
from donaldson import donaldson, initial_balanced_metric
from metric_measures import sigma_error, metric_point_weight_dtype

# training point, determinant  

# - determinant - optimal reproduce for given k 
# - metric - for batches 

# pull-back metric - what are our features? point weights
#   what are our labels? inner product of two points? or is it the metric itself?
#   

#input size of COORDINATES
COORDINATES = 5

def config_model():
    model = keras.Sequential()
    model.add(layers.Dense(32, activation="relu", input_shape=(COORDINATES,)))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))
    return model

def sigma_error_loss(x_true, sample_size):
    if sample_size > len(x_true):
        raise ValueError("sample_size must not exceed length of x_true argument" )

    sigma_error_vec = np.vectorize(sigma_error, signature='(m)->()')
    sigma_for_pairs = lambda sample_pairs : np.sum( sigma_error_vec(sample_pairs), axis=1)  
    return lambda y_true, y_pred : sigma_for_pairs(sample_point_pairs(metric_point_weights(y_pred, x_true), sample_size))

def metric_point_weights(metrics, point_weights):
    g_point_weights = np.zeros(len(point_weights), dtype=metric_point_weight_dtype)
    g_point_weights ['point_weight'] = point_weights
    g_point_weights['metric'] = metrics
    return g_point_weights

def sample_point_pairs(metric_point_weights, sample_size):
    batch_size = len(metric_point_weights)
    exclude = lambda x, arr : list(filter(lambda p : np.array_equal(p, x) == False, arr))
    sample_indices = lambda size, sample_size : np.random.choice(size, sample_size, replace=False)
    sample_with_exclude = lambda x, arr : np.take(exclude(x, arr), sample_indices(batch_size - 1, sample_size), axis=0)

    samples = [ np.array( list(map(lambda y : np.array([x, y]), sample_with_exclude(x, metric_point_weights))) ) 
        for x in metric_point_weights ]
    return np.array(samples)

if __name__ == '__main__':
    y_pred = [ initial_balanced_metric(3) for _ in range(10) ]
    x_true = fq.quintic_point_weights(10)
    sample_size= 4
    print(sigma_error_loss(x_true, sample_size)(None, y_pred))
