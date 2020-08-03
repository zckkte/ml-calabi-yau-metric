from time import time
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tf_metric_measures 
import fermat_quintic as fq
from donaldson import donaldson, initial_balanced_metric

COORDINATES = 5
LEARNING_RATE = 1e-4

def main(epochs=3, batch_size=32, sample_size = 4, no_of_samples = 10000):
    model = config_model()
    model.compile(optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE), metrics=['accuracy'])

    features = convert_to_ndarray(fq.quintic_point_weights(no_of_samples))
    train_dataset = tf.data.Dataset.from_tensor_slices((features, features))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
    model = model_train(model, train_dataset, sample_size, epochs)

    model.save('%d_model_b%d_s%d_n%d.h5' % (int(time()), batch_size, sample_size, no_of_samples))

convert_to_ndarray = (lambda point_weights : 
    np.array(list(map(lambda pw : np.append(pw['point'].view(np.float32), pw['weight']), point_weights)), dtype=np.float32))

def model_train(model, train_dataset, sample_size=4, epochs=5):
    for epoch in range(1, epochs + 1):
        print("epoch %d/%d" % (epoch, epochs))
        for _, (x_batch_train, _) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                logits = model(x_batch_train, training=True)
                losses = sigma_loss(sample_size)(x_batch_train, logits)
            grads = tape.gradient(losses, model.trainable_weights)
            model.optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return model

def config_model():
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(2 * COORDINATES + 1, )))
    model.add(keras.layers.Lambda(pad_input))
    model.add(keras.layers.Dense(32, activation='relu'))
    model.add(keras.layers.Dense(64, activation="relu"))
    model.add(keras.layers.Dense(128, activation="relu"))
    model.add(keras.layers.Dense((COORDINATES * 2) + 2))
    return model

pad_input = lambda x : tf.map_fn(lambda y : pad_to_output_dim(y,(COORDINATES * 2) + 2), x)

pad_to_output_dim = lambda t, out_dim : tf.pad(t, [[0, tf.abs(out_dim - t.shape[0]) ]], "CONSTANT")

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

    #HACK: abusing y_true argument for the purpose of providing x_true as input to loss function
    return lambda x_true, y_pred : \
        sigma(sample_tuples(concat_point_weight_det(x_true[:, 0:2 * COORDINATES + 1], \
            extract_y_pred_to_hermitian(y_pred)), sample_size))

#HACK: keras does not allow for y_true and y_pred to be of differring shapes 
extract_y_pred_to_hermitian = lambda y_pred: tf.map_fn(lambda y : to_hermitian(y), y_pred, dtype=tf.complex64)

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

if __name__ == '__main__':
    main()
