import os.path

import tensorflow as tf

class LengthTTVLossLayer(tf.keras.layers.Layer):
    def __init__(self, active_huber_weight=1.0, blank_huber_weight=1.0, kl_weight=0.1):
        self.active_huber_weight = tf.convert_to_tensor(active_huber_weight, dtype=tf.float32)
        self.blank_huber_weight = tf.convert_to_tensor(blank_huber_weight, dtype=tf.float32)
        self.kl_weight = tf.convert_to_tensor(kl_weight, dtype=tf.float32)
        super(LengthTTVLossLayer, self).__init__()
    def call(self, inputs, *args, **kwargs):
        cam_output, count, length = inputs

        batch_size = tf.shape(cam_output)[0]
        max_length = tf.shape(cam_output)[1]

        def process_batch(i):
            active_pred = cam_output[i, :length[i], ...]
            blank_pred = cam_output[i, length[i]:, ...]
            active_uniform_distribution = tf.divide(tf.ones_like(active_pred), tf.cast(tf.shape(active_pred)[1], tf.float32))
            active_pred = tf.pad(active_pred, [[0, max_length - tf.shape(active_pred)[0]], [0, 0], [0, 0], [0, 0]])
            active_uniform_distribution = tf.pad(active_uniform_distribution, [[0, max_length - tf.shape(active_uniform_distribution)[0]], [0, 0], [0, 0], [0, 0]])
            blank_pred = tf.pad(blank_pred, [[0, max_length - tf.shape(blank_pred)[0]], [0, 0], [0, 0], [0, 0]])
            return active_pred, active_uniform_distribution, blank_pred

        active_predictions, active_uniform_distributions, blank_predictions = tf.map_fn(
            process_batch,
            elems=tf.range(batch_size),
            dtype=(cam_output.dtype, cam_output.dtype, cam_output.dtype),
            parallel_iterations=32
        )

        active_huber_loss = tf.losses.huber(count, tf.reduce_sum(active_predictions, axis=[1, 2, 3])) * self.active_huber_weight
        blank_huber_loss = tf.losses.huber(0, tf.reduce_sum(blank_predictions, axis=[1, 2, 3])) * self.blank_huber_weight
        kl_loss = tf.keras.losses.KLDivergence()(active_uniform_distributions, active_predictions) * self.kl_weight
        total_loss = active_huber_loss + blank_huber_loss + kl_loss
        self.add_loss(total_loss)
        self.add_metric(active_huber_loss, name='active_huber_loss')
        self.add_metric(blank_huber_loss, name='blank_huber_loss')
        self.add_metric(kl_loss, name='kl_loss')

        return cam_output

def prepare(class_num, image_size=320, bottle_neck=64, pretrain_weight_path=None, pretrain_freeze=False, fine=False):
    base_model = tf.keras.applications.MobileNetV2(weights='imagenet' if fine else None,
                                                   include_top=False, input_shape=(image_size, image_size, 3))
    base_output = tf.keras.layers.Conv2D(filters=bottle_neck, kernel_size=3, activation='relu', padding='same')(base_model.layers[118].output)
    partial_model = tf.keras.Model(inputs=base_model.input, outputs=base_output)
    if pretrain_weight_path is not None:
        load_status = partial_model.load_weights(pretrain_weight_path)
        if pretrain_freeze and load_status:
            for layer in partial_model.layers:
                layer.trainable = False

    inputs = tf.keras.Input(shape=(None, image_size, image_size, 3))
    count = tf.keras.layers.Input((class_num), dtype=tf.int32)
    length = tf.keras.layers.Input((), dtype=tf.int32)
    x0 = tf.keras.layers.TimeDistributed(partial_model)(inputs)
    x0 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(filters=bottle_neck, kernel_size=1, activation='relu', padding='same'))(x0)
    x0 = tf.keras.layers.Conv3D(filters=bottle_neck//2, kernel_size=(3, 3, 3), dilation_rate=(1, 1, 1), activation='relu', padding='same')(x0)
    x0 = tf.keras.layers.Conv3D(filters=bottle_neck//2, kernel_size=(3, 3, 3), dilation_rate=(2, 2, 2), activation='relu', padding='same')(x0)
    x0 = tf.keras.layers.Conv3D(filters=bottle_neck//2, kernel_size=(3, 3, 3), dilation_rate=(4, 4, 4), activation='relu', padding='same')(x0)
    x0 = tf.keras.layers.Conv3D(filters=bottle_neck//2, kernel_size=(3, 3, 3), dilation_rate=(8, 8, 8), activation='relu', padding='same')(x0)
    # Block 3
    ## Resize
    cam_output = tf.keras.layers.Conv3D(filters=class_num, kernel_size=(3, 3, 3), dilation_rate=(1, 1, 1), activation='relu', padding='same')(x0)
    predictions = LengthTTVLossLayer()([cam_output, count, length])
    count_output = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=[1, 2, 3]))(cam_output)
    train_model = tf.keras.Model(inputs=[inputs, count, length], outputs=predictions)
    save_model = tf.keras.Model(inputs=inputs, outputs=[count_output, cam_output])
    return train_model, save_model