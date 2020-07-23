import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import fermat_quintic as fq
from donaldson import donaldson, initial_balanced_metric
from metric_measures import sigma_error, metric_point_weight_dtype
import tf_metric_measures 

# training point, determinant  

# - determinant - optimal reproduce for given k 
# - metric - for batches 

# pull-back metric - what are our features? point weights
#   what are our labels? inner product of two points? or is it the metric itself?
#   

#input size of COORDINATES
COORDINATES = 5
LEARNING_RATE = 1e-4

def main():
    batch_size=100
    model = config_model(batch_size=batch_size)
    features = convert_to_ndarray(fq.quintic_point_weights(10000))
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer = optimizer,loss=sigma_loss(sample_size=80), metrics=['accuracy'])
    model.fit(x=features, y=features, batch_size=batch_size, epochs=5, verbose=True)

convert_to_ndarray = (lambda point_weights : 
    np.array(list(map(lambda pw : np.append(pw['point'].view(np.float32), pw['weight']), point_weights)) ))

def generator(features, labels, batch_size):
    while True:
        indices = np.random.choice(len(features), batch_size)
        batch_features = np.take(features, indices)
        yield batch_features, None

def config_model(batch_size):
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(2 * COORDINATES + 1, )))
    model.add(keras.layers.Dense(32, activation='relu'))
    model.add(keras.layers.Dense(64, activation="relu"))
    model.add(keras.layers.Dense(128, activation="relu"))
    model.add(keras.layers.Dense((COORDINATES * 2) + 2))
    model.add(keras.layers.Lambda(lambda batch : tf.map_fn(lambda y : to_hermitian(y), batch) )) #issue here unresolved
    return model

def fill_triangular(x):
    m = x.shape[0]
    n = tf.cast(tf.math.sqrt(.25 + 2 * m) - .5, tf.int32)
    x_tail = x[(m - (n**2 - m)):]
    x = tf.reshape( tf.concat([x, x_tail[::-1]], 0), (n, n))
    return tf.linalg.band_part(x, 0, -1)

def to_hermitian(x):
    real, imag = x[::2], x[1::2]
    values = tf.complex(real, imag) 
    dig_upper = fill_triangular(values)
    lower_tri = tf.transpose(tf.linalg.band_part(dig_upper, 0, -1), conjugate=True) 
    return dig_upper + lower_tri

def sigma_loss(sample_size): 
    sigma_error_vec = lambda pairs : tf.foldl (lambda sigma_acc, pair : sigma_acc 
        + tf_metric_measures.sigma_error(pair), pairs, 0.)
    sigma= lambda sample_pairs : tf.map_fn( lambda pairs : sigma_error_vec(pairs), sample_pairs, dtype=tf.float32)  
    return lambda x_true, y_pred : sigma(sample_tuples(concat_point_weight_det(x_true, y_pred), sample_size))

def to_complex_point_weight(x_true):
    point_real = x_true[:, 0:COORDINATES * 2][:, ::2]
    point_imag = x_true[:, 0:COORDINATES * 2][:, 1::2]
    point_complex = tf.complex(point_real, point_imag)
    weights = x_true[:, -1]
    return tf.concat([point_complex, cast_expand_dim (weights)], axis=1)

def concat_point_weight_det(x_true, metrics):
    determinants = tf.linalg.det(metrics)
    return tf.concat([ to_complex_point_weight(x_true), cast_expand_dim(determinants) ], axis=1)

cast_expand_dim = lambda arr : tf.cast(tf.expand_dims(arr, 1), dtype=tf.complex64)

def sample_tuples(metric_point_weights, sample_size):
    exclude_mask = (lambda x, arr : tf.map_fn(lambda p : tf.reduce_all( tf.equal(p, x) ) == False,  
        arr, dtype=tf.bool))
    exclude = lambda x, arr : tf.boolean_mask(arr, exclude_mask(x, arr))
    sample_with_exclude = lambda x, arr : random_choice(exclude(x, arr), sample_size, axis=0) 
    stack_map = lambda x, arr : tf.map_fn(lambda y : tf.stack([x, y]), arr)

    return tf.map_fn(lambda x : stack_map(x, sample_with_exclude(x, metric_point_weights)), metric_point_weights)

def random_choice(x, size, axis=0, unique=True):
    dim_x = tf.cast(tf.shape(x)[axis], tf.int64)
    indices = tf.range(0, dim_x, dtype=tf.int64)
    sample_index = tf.random.shuffle(indices)[:size]
    return tf.gather(x, sample_index, axis=axis)

def __main_test():
    y_pred = tf.convert_to_tensor([ initial_balanced_metric(3) for _ in range(10) ])
    x_true = tf.convert_to_tensor(convert_to_ndarray(fq.quintic_point_weights(10)))
    sample_size=4
    print(sigma_loss(sample_size)(x_true, y_pred))

if __name__ == '__main__':
    main()
