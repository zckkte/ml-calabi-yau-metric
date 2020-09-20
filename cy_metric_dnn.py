import argparse
from time import time
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tf_metric_measures 
import fermat_quintic as fq
from donaldson import donaldson, initial_balanced_metric

COORDINATES = 5
LEARNING_RATE = 1e-4

def main(epochs=3, batch_size=32, sample_size=28, no_of_samples = 10000):
    model = config_model()
    model.compile(optimizer = keras.optimizers.Adagrad(learning_rate=LEARNING_RATE), metrics=['accuracy'])

    features = convert_to_ndarray(fq.quintic_point_weights(no_of_samples))
    train_dataset = tf.data.Dataset.from_tensor_slices((features, features))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
    model, loss = model_train(model, train_dataset, batch_size, sample_size, epochs, loss_func=sigma_loss_alt)

    file_name = '%d_model_b%d_s%d_n%d.h5' % (int(time()), batch_size, sample_size, no_of_samples)
    test = convert_to_ndarray(fq.quintic_point_weights(int(0.3 * no_of_samples)))
    print('sigma-measure (train): %f' % eval_model_sigma(model, features).numpy())
    print('sigma-measure (test): %f' % eval_model_sigma(model, test).numpy())

    np.save(file_name, loss.numpy())
    model.save(file_name)

def eval_model_sigma(model, point_weights):
    gs = model.predict(point_weights)
    return tf_metric_measures.sigma_error(concat_point_weight_det(point_weights, gs))

convert_to_ndarray = (lambda point_weights : 
    np.array(list(map(lambda pw : np.append(pw['point'].view(np.float32), pw['weight']), point_weights)), dtype=np.float32))

sigma_loss_alt = tf.function(lambda x_true, y_pred : 
    tf_metric_measures.simple_sigma(concat_point_weight_det(x_true, y_pred)))

def sigma_loss(sample_size, batch_size): 
    sigma_error_vec = lambda pairs : tf.foldl (lambda sigma_acc, pair : sigma_acc 
        + tf_metric_measures.sigma_error(pair), pairs, 0., parallel_iterations=batch_size)
    sigma = lambda sample_pairs : tf.map_fn(sigma_error_vec, sample_pairs, dtype=tf.float32, 
        parallel_iterations=sample_pairs.shape[0])

    #HACK: abusing y_true argument for the purpose of providing x_true as input to loss function
    return tf.function(lambda x_true, y_pred : sigma(sample_tuples(concat_point_weight_det(x_true, y_pred), sample_size)) )

def model_train(model, train_dataset, batch_size, sample_size, epochs=5, loss_func=sigma_loss_alt):
    loss_history = []
    for epoch in tf.range(1, epochs + 1):
        print("epoch %d/%d" % (epoch, epochs))
        for step, (x_batch_train, _) in enumerate(train_dataset): 
            with tf.GradientTape() as tape:
                logits = model(x_batch_train, training=True)
                losses = loss_func(x_batch_train, logits)
            grads = tape.gradient(losses, model.trainable_weights)
            model.optimizer.apply_gradients(zip(grads, model.trainable_weights))
            loss_history.append(tf.reduce_sum(losses))
    return model, tf.concat(loss_history, axis=0)

def config_model():
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(2 * COORDINATES + 1, )))
    model.add(keras.layers.Dense(128, activation="relu"))
    model.add(keras.layers.Dense(64, activation="relu"))
    model.add(keras.layers.Dense(32, activation='relu'))
    model.add(keras.layers.Dense((COORDINATES * 2) - 1))
    model.add(keras.layers.Lambda(to_hermitian_batch))
    return model

to_hermitian_batch = tf.function(lambda y_pred : tf.vectorized_map(to_hermitian, y_pred))

def to_hermitian(x):
    t1 = tf.reshape(tf.complex(x, tf.zeros(9, dtype=tf.float32)), (3,3))
    up = tf.linalg.band_part(t1, 0, -1)
    low = tf.linalg.band_part(1j * t1, -1, 0)
    out = up + tf.transpose(up) - tf.linalg.band_part(t1, 0, 0)
    return out + low + tf.math.conj(tf.transpose(low))

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
    exclude_mask = lambda x, arr : tf.map_fn(lambda p : tf.equal(tf.reduce_all(tf.equal(p, x)), False), arr, dtype=tf.bool, 
        parallel_iterations=arr.shape[0])
    exclude = lambda x, arr : tf.boolean_mask(arr, exclude_mask(x, arr))

    sample_with_exclude = lambda x, arr : random_choice(exclude(x, arr), sample_size, axis=0)
    stack_map = lambda x, arr : tf.map_fn(lambda y : tf.stack([x, y]), arr, parallel_iterations=arr.shape[0])
    return tf.map_fn(lambda x : stack_map(x, sample_with_exclude(x, metric_point_weights)), metric_point_weights,
        parallel_iterations=metric_point_weights.shape[0])

def random_choice(x, size, axis=0):
    dim_x = tf.cast(tf.shape(x)[axis], tf.int64)
    indices = tf.range(0, dim_x, dtype=tf.int64)
    sample_index = tf.random.shuffle(indices)[:size]
    return tf.gather(x, sample_index, axis=axis)

def parser_config():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-e', type=int, required=True, help='number of epochs')
    parser.add_argument('-b', type=int, required=True, help='batch size')
    parser.add_argument('-s', type=int, default=0, help='loss sample size')
    parser.add_argument('-N', type=int, default=100000, help='number of sample points')
    return parser

if __name__ == '__main__':
    args = parser_config().parse_args()
    main(epochs=args.e, batch_size=args.b, sample_size=args.s, no_of_samples=args.N)
