# -*- coding: UTF-8 -*-
import keras
from keras.models import Model
import numpy as np
#如果精度在多余一轮的时间内不再改善 中断训练  filepath用于保存目标模型文件如model.h5 
#ModelCheckpoint monitor 如果val_loss没有改善，不需要覆盖模型
callbacks_list = [keras.callbacks.EarlyStopping(monitor='acc',patience=1)
,keras.callbacks.ModelCheckpoint(filepath='',monitor='val_loss',save_best_only=True)]

class ActivationLogger(keras.callbacks):
    def set_model(self,model):
        self.model = model
        layer_outputs = [layer.output for layer in model.layers]
        self.activations_model = keras.models.Model(model.input,layer_outputs)
    
    def on_epoch_end(self,epoch,logs=None):
        if self.validation_data is None:
            raise RuntimeError('Requires validation_data')
        validation_sample = self.validation_data[0][0:1]
        activations = self.activations_model.predict(validation_sample)
        f = open('activations_at_epoch_' + str(epoch) + '.npz','w')
        np

