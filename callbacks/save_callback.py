from PIL import Image
import os
import tensorflow as tf
from dump_train_dataset import create_gif

class SaveCallback(tf.keras.callbacks.Callback):
    def __init__(self, save_model, save_model_dir_path, prefix, valid_data, train_valid_data, logging_sample=16):
        super(SaveCallback, self).__init__()
        os.makedirs(save_model_dir_path, exist_ok=True)

        self.save_model_dir_path = save_model_dir_path
        self.prefix = prefix
        self.valid_data = valid_data
        self.train_valid_data = train_valid_data
        self.logging_sample = logging_sample
        self.save_model = save_model

    def on_epoch_end(self, epoch, logs=None):
        loss_string = "_".join([f'{k}_{v:.4f}' for k, v in logs.items()])
        save_model_name = f'{self.prefix}_epoch-{epoch}_{loss_string}.h5'
        output_model_file_path = os.path.join(self.save_model_dir_path, save_model_name)
        os.makedirs(os.path.dirname(output_model_file_path), exist_ok=True)
        self.save_model.save(output_model_file_path)

        self.predict_dump(output_model_file_path + '_log_train', self.train_valid_data)
        self.predict_dump(output_model_file_path + '_log_valid', self.valid_data)

    def predict_dump(self, output_log_dir_path, valid_data):
        os.makedirs(output_log_dir_path, exist_ok=True)
        result = self.model.predict(valid_data[0][:self.logging_sample])
        for index, video in enumerate(valid_data[0][:self.logging_sample]):
            answer_string = ""
            for answer in valid_data[1][index]:
                answer_string += f'{int(answer)}_'
            answer_string = answer_string[:-1]
            inf_string = ""
            for inf in result[index]:
                inf_string += f'{round(inf)}_'
            inf_string = inf_string[:-1]
            output_file_path = os.path.join(output_log_dir_path, f'{index:04d}_ans-{answer_string}_inf-{inf_string}.gif')
            create_gif(video, output_file_path)