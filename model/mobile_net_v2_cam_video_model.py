import os.path

import tensorflow as tf

def prepare(class_num, image_size=320, bottle_neck=64, pretrain_model_path=None, pretrain_freeze=False, fine=False):
    base_model = tf.keras.applications.MobileNetV2(weights='imagenet' if fine else None,
                                                   include_top=False, input_shape=(image_size, image_size, 3))
    base_output = tf.keras.layers.Conv2D(filters=bottle_neck, kernel_size=3, activation='relu', padding='same')(base_model.layers[118].output)
    partial_model = tf.keras.Model(inputs=base_model.input, outputs=base_output)
    if pretrain_model_path is not None:
        load_status = partial_model.load_weights(pretrain_model_path, skip_mismatch=True, by_name=True)
        if pretrain_freeze:
            for layer in partial_model.layers:
                layer.trainable = False

    inputs = tf.keras.Input(shape=(None, image_size, image_size, 3))
    x0 = tf.keras.layers.TimeDistributed(partial_model)(inputs)
    x0 = tf.keras.layers.Conv3D(filters=bottle_neck//2, kernel_size=(3, 3, 3), dilation_rate=(1, 1, 1), activation='relu', padding='same')(x0)
    x0 = tf.keras.layers.Conv3D(filters=bottle_neck//2, kernel_size=(3, 3, 3), dilation_rate=(2, 2, 2), activation='relu', padding='same')(x0)
    x0 = tf.keras.layers.Conv3D(filters=bottle_neck//2, kernel_size=(3, 3, 3), dilation_rate=(4, 4, 4), activation='relu', padding='same')(x0)
    x0 = tf.keras.layers.Conv3D(filters=bottle_neck//2, kernel_size=(3, 3, 3), dilation_rate=(8, 8, 8), activation='relu', padding='same')(x0)
    # Block 3
    ## Resize
    cam_output = tf.keras.layers.Conv3D(filters=class_num, kernel_size=(3, 3, 3), dilation_rate=(1, 1, 1), activation='relu', padding='same')(x0)
    predictions = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=[1, 2, 3]))(cam_output)
    train_model = tf.keras.Model(inputs=inputs, outputs=predictions)
    save_model = tf.keras.Model(inputs=inputs, outputs=[predictions, cam_output])
    return train_model, save_model