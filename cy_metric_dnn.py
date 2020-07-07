import numpy as np
import keras
from keras import layers
import keras.backend as kb
import fermat_quintic as fq
from donaldson import donaldson
from metric_measures import sigma_error

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

    point_sample_tuples = sample_point_pairs(x_true, sample_size)
    sigma_error_vec = np.vectorize(sigma_error, signature='(m,m),()->()')
    sigma_pair = lambda g_pb, point_weight_pairs : sum( sigma_error_vec(g_pb, point_weight_pairs) ) 
    return lambda y_true, y_pred : np.array([ sigma_pair(g_pb, point_weight_pairs) 
        for g_pb, point_weight_pairs in zip(y_pred, point_sample_tuples) ])

def sample_point_pairs(x_true, sample_size):
    batch_size = len(x_true)
    exclude = lambda x, arr : np.array(list(filter(lambda p : np.array_equal(p, x) == False, arr)))
    sample_indices = lambda batch_size : np.random.choice(batch_size - 1, sample_size, replace=False)

    sample_with_exclude = lambda x, arr : np.take(exclude(x, arr), sample_indices(batch_size), axis=0)
    sample_tuples = [ np.array( list(map(lambda y : np.array([x, y]), sample_with_exclude(x, x_true))) ) 
        for x in x_true ]
    return np.array(sample_tuples)

if __name__ == '__main__':
    k=4
    points = fq.sample_quintic_points(10000)
    h_bal = donaldson(k, max_iterations=15)
    pull_back_det_vec = np.vectorize(fq.pull_back_determinant)
    g_det = pull_back_det_vec(k, h_bal, points)

    model = config_model()
    model.compile(optimizer='rmsprop', loss=[sigma_error_loss], metrics=['accuracy'])
    model.fit(points, g_det)
