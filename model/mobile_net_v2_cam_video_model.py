import tensorflow as tf

def prepare(class_num, image_size=320, bottle_neck=64, fine=False):
    base_model = tf.keras.applications.MobileNetV2(weights='imagenet' if fine else None,
                                                   include_top=False, input_shape=(image_size, image_size, 3))
    base_output = tf.keras.layers.Conv2D(filters=bottle_neck, kernel_size=3, activation='relu', padding='same')(base_model.layers[118].output)
    partial_model = tf.keras.Model(inputs=base_model.input, outputs=base_output)

    inputs = tf.keras.Input(shape=(None, image_size, image_size, 3))
    x0 = tf.keras.layers.TimeDistributed(partial_model)(inputs)

    # Block 1
    ## Resize
    x0 = tf.keras.layers.Conv3D(filters=bottle_neck//4, kernel_size=(3, 3, 3), strides=(2, 1, 1), padding='same')(x0)
    ## ResConv2Plus1D
    x1 = tf.keras.layers.Conv3D(filters=bottle_neck//4, kernel_size=(1, 7, 7), padding='same')(x0)
    x1 = tf.keras.layers.Conv3D(filters=bottle_neck//4, kernel_size=(3, 1, 1), padding='same')(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.ReLU()(x1)
    x1 = tf.keras.layers.Conv3D(filters=bottle_neck//4, kernel_size=(1, 7, 7), padding='same')(x1)
    x1 = tf.keras.layers.Conv3D(filters=bottle_neck//4, kernel_size=(3, 1, 1), padding='same')(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.Add()([x0, x1])
    # Block 2
    ## Resize
    x1 = tf.keras.layers.Conv3D(filters=bottle_neck//2, kernel_size=(3, 3, 3), strides=(2, 1, 1), padding='same')(x1)
    ## ResConv2Plus1D
    x2 = tf.keras.layers.Conv3D(filters=bottle_neck//2, kernel_size=(1, 7, 7), padding='same')(x1)
    x2 = tf.keras.layers.Conv3D(filters=bottle_neck//2, kernel_size=(3, 1, 1), padding='same')(x2)
    x2 = tf.keras.layers.BatchNormalization()(x2)
    x2 = tf.keras.layers.ReLU()(x2)
    x2 = tf.keras.layers.Conv3D(filters=bottle_neck//2, kernel_size=(1, 7, 7), padding='same')(x2)
    x2 = tf.keras.layers.Conv3D(filters=bottle_neck//2, kernel_size=(3, 1, 1), padding='same')(x2)
    x2 = tf.keras.layers.BatchNormalization()(x2)
    x11 = tf.keras.layers.Dense(units=bottle_neck//2)(x1)
    x11 = tf.keras.layers.BatchNormalization()(x11)
    x2 = tf.keras.layers.Add()([x11, x2])
    # Block 3
    ## Resize
    cam_output = tf.keras.layers.Conv3D(filters=class_num, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x2)
    predictions = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=[1, 2, 3]))(cam_output)
    train_model = tf.keras.Model(inputs=inputs, outputs=predictions)
    save_model = tf.keras.Model(inputs=inputs, outputs=[predictions, cam_output])
    return train_model, save_model