import tensorflow as tf

def ttv_huber_loss(weight=0.01):
    huber_loss = tf.keras.losses.Huber()
    def loss(y_true, y_pred):
        basic_loss = huber_loss(y_true, y_pred)
        frame_diff = y_pred[:, 1:] - y_pred[:, :-1]
        variance = tf.math.reduce_variance(frame_diff)
        variance_loss = 1 / (variance + 1e-6)
        return basic_loss + weight * variance_loss
    return loss
