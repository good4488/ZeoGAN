import math
import functools

import tensorflow as tf

kernel_initializer = tf.random_normal_initializer(0.0, 0.02)


dense = functools.partial(
    tf.layers.dense,
    activation=None,
    use_bias=False,
    kernel_initializer=kernel_initializer,
)

def smooth_square(x, upper, lower):
    return tf.where(x>upper, x-upper, tf.zeros_like(x))**2 \
         + tf.where(x<lower, x-lower, tf.zeros_like(x))**2


def pbc_pad3d(x, lp, rp, name="PBC"):
    with tf.variable_scope(name):
        if lp == 0 and rp == 0:
            x = tf.identity(x)
        elif lp == 0 and rp != 0:
            x = tf.concat([x, x[:, :rp, :, :, :]], axis=1)
            x = tf.concat([x, x[:, :, :rp, :, :]], axis=2)
            x = tf.concat([x, x[:, :, :, :rp, :]], axis=3)
        elif lp != 0 and rp != 0:
            x = tf.concat(
                [x[:, -lp:, :, :, :], x, x[:, :rp, :, :, :]], axis=1)
            x = tf.concat(
                [x[:, :, -lp:, :, :], x, x[:, :, :rp, :, :]], axis=2)
            x = tf.concat(
                [x[:, :, :, -lp:, :], x, x[:, :, :, :rp, :]], axis=3)
        else:
            raise Exception("lp != 0 and rp == 0")

    return x


def pbc_conv3d(x, pbc=True, **kwargs):
    if pbc:
        # Calculate padding size.
        s = kwargs["strides"]
        k = kwargs["kernel_size"]

        # i = input size.
        i = x.get_shape().as_list()[1]

        if i % s == 0:
            p = max(k-s, 0)
        else:
            p = max(k - (i%s), 0)

        # calc left padding = lp and right padding = rp
        lp = p // 2
        rp = p - lp

        # Pad.
        x = pbc_pad3d(x, lp, rp)

        kwargs["padding"] = "VALID"

    # Do convolution.
    x = tf.layers.conv3d(x, **kwargs)

    return x


conv3d = functools.partial(
    pbc_conv3d,
    pbc=True,
    kernel_size=5,
    strides=2,
    padding="SAME",
    activation=None,
    use_bias=False,
    kernel_initializer=kernel_initializer,
)


conv3d_transpose = functools.partial(
    tf.layers.conv3d_transpose,
    kernel_size=5,
    strides=2,
    padding="SAME",
    activation=None,
    use_bias=False,
    kernel_initializer=kernel_initializer,
)

# Source: https://github.com/maxorange/voxel-dcgan/blob/master/ops.py
# Automatic updator version of batch normalization.
def batch_normalization(
        x,
        training,
        name="batch_normalization",
        decay=0.99,
        epsilon=1e-5,
        global_norm=True):
    # Get input shape as python list.
    shape = x.get_shape().as_list()

    if global_norm:
        # Channel-wise statistics.
        size = shape[-1:]
        axes = list(range(len(shape)-1))
        keep_dims = False
    else:
        # Pixel-wise statistics.
        size = [1] + shape[1:]
        axes = [0]
        keep_dims = True

    with tf.variable_scope(name):
        beta = tf.get_variable(
            name="beta",
            shape=size,
            initializer=tf.constant_initializer(0.0),
        )
        gamma = tf.get_variable(
            name="gamma",
            shape=size,
            initializer=tf.random_normal_initializer(1.0, 0.02),
        )
        moving_mean = tf.get_variable(
            name="moving_mean",
            shape=size,
            initializer=tf.constant_initializer(0.0),
            trainable=False,
        )
        moving_var = tf.get_variable(
            name="moving_var",
            shape=size,
            initializer=tf.constant_initializer(1.0),
            trainable=False,
        )

        # Add moving vars to the tf collection.
        # The list of moving vars can be obtained with
        # tf.moving_average_variables().
        if moving_mean not in tf.moving_average_variables():
            collection = tf.GraphKeys.MOVING_AVERAGE_VARIABLES
            tf.add_to_collection(collection, moving_mean)
            tf.add_to_collection(collection, moving_var)

        def train_mode():
            # execute at training time
            batch_mean, batch_var = tf.nn.moments(
                                        x,
                                        axes=axes,
                                        keep_dims=keep_dims,
                                    )
            update_mean = tf.assign_sub(
                moving_mean, (1-decay) * (moving_mean-batch_mean)
            )
            update_var = tf.assign_sub(
                moving_var, (1-decay) * (moving_var-batch_var)
            )

            # Automatically update global means and variances.
            with tf.control_dependencies([update_mean, update_var]):
                return tf.nn.batch_normalization(
                            x, batch_mean, batch_var, beta, gamma, epsilon)

        def test_mode():
            # execute at test time
            return tf.nn.batch_normalization(
                       x, moving_mean, moving_var, beta, gamma, epsilon)

        return tf.cond(training, train_mode, test_mode)


def minibatch_discrimination(x, num_kernels, dim_per_kernel, name="minibatch"):
    input_x = x

    with tf.variable_scope(name):
        x = dense(x, units=num_kernels*dim_per_kernel)
        x = tf.reshape(x, [-1, num_kernels, dim_per_kernel])

        diffs = (
            tf.expand_dims(x, axis=-1) -
            tf.expand_dims(tf.transpose(x, [1, 2, 0]), axis=0)
        )

        l1_dists = tf.reduce_sum(tf.abs(diffs), axis=2)
        minibatch_features = tf.reduce_sum(tf.exp(-l1_dists), axis=2)

        return tf.concat([input_x, minibatch_features], axis=1)


def match_shape_with_dense(x, target, name="match_shape"):
    # Get shape and replace None to -1.
    shape = [i if i else -1 for i in target.get_shape().as_list()]

    flat_size = 1
    for s in shape[1:]:
        flat_size *= s

    with tf.variable_scope(name):
        x = tf.layers.flatten(x)
        x = dense(x, units=flat_size, use_bias=True)

        # Same size as input
        x = tf.reshape(x, shape=shape)

    return x


if __name__ == "__main__":
    import numpy as np
    data = np.fromfile("/home/FRAC32/RWY/RWY.griddata", dtype=np.float32)
    data = data.reshape([1, 32, 32, 32, 1])
    v = tf.Variable(data)
    v = pbc_pad3d(v, 22, 15)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        data = sess.run(v)

    data.tofile("test.times")
