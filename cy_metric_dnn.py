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
    model = config_model()
    point_weights = fq.quintic_point_weights(10000)
    model.fit_generator(generator=generator(point_weights, point_weights, batch_size=4), 
        validation_data=None, steps_per_epoch=None, epochs=5, verbose=True)
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer = optimizer,loss=sigma_error_loss, metrics=['accuracy', 'loss'])

def generator(features, labels, batch_size):
    pass

def config_model():
    return keras.Sequential([ 
            keras.layers.Dense(100, activation="relu", input=(COORDINATES+1,)),
            keras.layers.Dense(100, activation="relu"),
            keras.layers.Dense(50, activation="relu"),
            keras.layers.Dense((COORDINATES - 2) ** 2)
        ])

def sigma_error_loss(x_true, sample_size): 
    if sample_size > len(x_true):
        raise ValueError("sample_size must not exceed length of x_true argument" )

    sigma_error_vec = lambda pairs : tf.foldl (lambda sigma_acc, pair : sigma_acc + tf_metric_measures.sigma_error(pair), pairs, 0.)
    sigma= lambda sample_pairs : tf.map_fn( lambda pairs : sigma_error_vec(pairs), sample_pairs, dtype=tf.float64)  
    return lambda y_true, y_pred : sigma(sample_point_pairs(concat_point_weight_det(x_true, y_pred), sample_size))

def concat_point_weight_det(point_weights, metrics):
    determinants = tf.linalg.det(metrics)
    return tf.concat([ point_weights, tf.expand_dims(determinants, 1)], axis=1)

def sample_point_pairs(metric_point_weights, sample_size):
    batch_size = len(metric_point_weights)
    exclude = lambda x, arr : list(filter(lambda p : tf.reduce_all( tf.equal(p, x) ) == False, arr) )
    sample_with_exclude = lambda x, arr : random_choice(exclude(x, arr), sample_size, axis=0) 
    samples = [ tf.convert_to_tensor( list(map(lambda y : np.array([x, y]), sample_with_exclude(x, metric_point_weights))) ) 
        for x in metric_point_weights ]
    return tf.convert_to_tensor(samples)

def random_choice(x, size, axis=0, unique=True):
    dim_x = tf.cast(tf.shape(x)[axis], tf.int64)
    indices = tf.range(0, dim_x, dtype=tf.int64)
    sample_index = tf.random.shuffle(indices)[:size]
    sample = tf.gather(x, sample_index, axis=axis)
    return sample

if __name__ == '__main__':
    y_pred = tf.convert_to_tensor([ initial_balanced_metric(3) for _ in range(10) ])
    x_true = tf.convert_to_tensor(list(map(lambda pw : np.append(pw['point'], pw['weight']), 
        fq.quintic_point_weights(10))))
    sample_size=4
    print(sigma_error_loss(x_true, sample_size)(None, y_pred))