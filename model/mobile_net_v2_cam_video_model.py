import os.path

import tensorflow as tf

def prepare(class_num, image_size=320, bottle_neck=64, pretrain_model_path=None, pretrain_freeze=False, fine=False):
    base_model = tf.keras.applications.MobileNetV2(weights='imagenet' if fine else None,
                                                   include_top=False, input_shape=(image_size, image_size, 3))
    base_output = tf.keras.layers.Conv2D(filters=bottle_neck, kernel_size=3, activation='relu', padding='same')(base_model.layers[118].output)
    partial_model = tf.keras.Model(inputs=base_model.input, outputs=base_output)
    if pretrain_model_path is not None:
        partial_model.load_weights(pretrain_model_path, skip_mismatch=True, by_name=True)
        if pretrain_freeze:
            for layer in partial_model.layers:
                layer.trainable = False

    inputs = tf.keras.Input(shape=(None, image_size, image_size, 3))
    x0 = tf.keras.layers.TimeDistributed(partial_model)(inputs)
    x0 = tf.keras.layers.Conv3D(filters=bottle_neck//2, kernel_size=(3, 3, 3), dilation_rate=(1, 1, 1), activation='relu', padding='same', name='conv3d_0')(x0)
    x0 = tf.keras.layers.Conv3D(filters=bottle_neck//2, kernel_size=(3, 3, 3), dilation_rate=(2, 2, 2), activation='relu', padding='same', name='conv3d_1')(x0)
    x0 = tf.keras.layers.Conv3D(filters=bottle_neck//2, kernel_size=(3, 3, 3), dilation_rate=(4, 4, 4), activation='relu', padding='same', name='conv3d_2')(x0)
    x0 = tf.keras.layers.Conv3D(filters=bottle_neck//2, kernel_size=(3, 3, 3), dilation_rate=(8, 8, 8), activation='relu', padding='same', name='conv3d_3')(x0)
    # Block 3
    ## Resize
    cam_output = tf.keras.layers.Conv3D(filters=class_num, kernel_size=(3, 3, 3), dilation_rate=(1, 1, 1), activation='relu', padding='same')(x0)
    predictions = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=[1, 2, 3]))(cam_output)
    train_model = tf.keras.Model(inputs=inputs, outputs=predictions)
    save_model = tf.keras.Model(inputs=inputs, outputs=[predictions, cam_output])
    grad_cam_save_model = prepare_grad_cam_model(save_model)
    return train_model, grad_cam_save_model

def prepare_grad_cam_model(model, time_distributed_layer_name='time_distributed', time_layer_name_list=('Conv1', 'block_1_depthwise', 'block_3_depthwise', 'block_6_depthwise'),
                           layer_name_list=('conv3d_0', 'conv3d_1', 'conv3d_2', 'conv3d_3')):
    time_distributed_layer = model.get_layer(time_distributed_layer_name)
    internal_model = time_distributed_layer.layer
    new_time_distributed_output_list = []
    for layer_name in time_layer_name_list:
        new_output = internal_model.get_layer(layer_name).output
        new_time_distributed_output = tf.keras.layers.TimeDistributed(tf.keras.Model(inputs=internal_model.input, outputs=new_output))(model.input)
        new_time_distributed_output_list.append(new_time_distributed_output)
    for layer_name in layer_name_list:
        new_time_distributed_output_list.append(model.get_layer(layer_name).output)
    new_time_distributed_output_list.append(model.output)
    new_model = tf.keras.models.Model(inputs=model.input, outputs=new_time_distributed_output_list)
    return new_model