
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Conv1D, MaxPooling1D, LSTM, Dropout
from tensorflow.keras import Model, Input, Sequential
import tensorflow as tf
from tensorflow.keras import regularizers

#import tensorflow_addons as tfa

def CNN_model():
    input_shape = Input(shape=(64, 9, 1))
    tower_1 = Conv2D(filters=16, kernel_size=(5,9), padding='same', activation='relu')(input_shape)
    tower_1 = MaxPooling2D((3, 1), strides=(3, 1), padding='same')(tower_1)

    tower_2 = Conv2D(filters=16, kernel_size=(5,9), padding='same', activation='relu')(tower_1)
    tower_2 = MaxPooling2D((3, 1), strides=(3, 1), padding='same')(tower_2)

    tower_3 =  Conv2D(filters=16, kernel_size=(5,9), padding='same', activation='relu')(tower_2)
    tower_3 = MaxPooling2D((3, 1), strides=(3, 1), padding='same')(tower_3)

    merged = Flatten()(tower_3)

    out = Dense(100, activation='relu')(merged)
    out = Dense(10, activation='relu')(out)
    out = Dense(1, activation='sigmoid')(out)

    model = Model(input_shape, out)
    return model

def CNN_model_1D(output_bias = None):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    input_shape = Input(shape=(128, 9, 1))
    tower_1 = Conv1D(filters=16, kernel_size=(5), padding='same', activation='relu')(input_shape)
    tower_1 = MaxPooling2D((2, 1), strides=(2, 1), padding='same')(tower_1)

    tower_2 = Conv1D(filters=16, kernel_size=(5), padding='same', activation='relu')(tower_1)
    tower_2 = MaxPooling2D((2, 1), strides=(2, 1), padding='same')(tower_2)

    tower_3 =  Conv1D(filters=16, kernel_size=(5), padding='same', activation='relu')(tower_2)
    tower_3 = MaxPooling2D((2, 1), strides=(2, 1), padding='same')(tower_3)

    merged = Flatten()(tower_3)

    out = Dense(100, activation='relu')(merged)
    out = Dropout(0.3)(out)
    out = Dense(10, activation='relu')(out)
    out = Dense(1, activation='sigmoid', bias_initializer=output_bias)(out)

    model = Model(input_shape, out)
    return model

def CNN_model_2D_simple(output_bias = None):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    input_shape = Input(shape=(128, 9, 1))
    tower_1 = Conv2D(filters=16, kernel_size=(5,1), padding='same', activation='relu')(input_shape)
    tower_1 = MaxPooling2D((2, 1), strides=(2, 1), padding='same')(tower_1)

    tower_2 = Conv2D(filters=16, kernel_size=(5,1), padding='same', activation='relu')(tower_1)
    tower_2 = MaxPooling2D((2, 1), strides=(2, 1), padding='same')(tower_2)

    tower_3 =  Conv2D(filters=16, kernel_size=(5,1), padding='same', activation='relu')(tower_2)
    tower_3 = MaxPooling2D((2, 1), strides=(2, 1), padding='same')(tower_3)

    merged = Flatten()(tower_3)

    out = Dense(50, activation='relu')(merged)
    out = Dropout(0.1)(out)
    out = Dense(10, activation='relu')(out)
    out = Dense(1, activation='sigmoid', bias_initializer=output_bias)(out)

    model = Model(input_shape, out)
    return model

def CNN_model_ensemble(output_bias = None):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    input_shape = Input(shape=(128, 3, 1))
    tower_1 = Conv2D(filters=16, kernel_size=(5,1), padding='same', activation='relu')(input_shape)
    tower_1 = MaxPooling2D((2, 1), strides=(2, 1), padding='same')(tower_1)

    tower_2 = Conv2D(filters=16, kernel_size=(5,1), padding='same', activation='relu')(tower_1)
    tower_2 = MaxPooling2D((2, 1), strides=(2, 1), padding='same')(tower_2)

    tower_3 =  Conv2D(filters=12, kernel_size=(5,1), padding='same', activation='relu')(tower_2)
    tower_3 = MaxPooling2D((2, 1), strides=(2, 1), padding='same')(tower_3)

    merged = Flatten()(tower_3)

    out = Dense(30, activation='relu')(merged)
    out = Dropout(0.1)(out)
    out = Dense(10, activation='relu')(out)
    out = Dense(1, activation='sigmoid', bias_initializer=output_bias)(out)

    model = Model(input_shape, out)
    return model

def CNN_model_ensemble_more_dropout(output_bias = None):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    input_shape = Input(shape=(128, 3, 1))
    tower_1 = Conv2D(filters=16, kernel_size=(5,1), padding='same', activation='relu')(input_shape)
    tower_1 = MaxPooling2D((2, 1), strides=(2, 1), padding='same')(tower_1)

    tower_2 = Conv2D(filters=16, kernel_size=(5,1), padding='same', activation='relu')(tower_1)
    tower_2 = MaxPooling2D((2, 1), strides=(2, 1), padding='same')(tower_2)
    tower_2 = Dropout(0.1)(tower_2)

    tower_3 =  Conv2D(filters=12, kernel_size=(5,1), padding='same', activation='relu')(tower_2)
    tower_3 = MaxPooling2D((2, 1), strides=(2, 1), padding='same')(tower_3)
    tower_3 = Dropout(0.1)(tower_3)

    merged = Flatten()(tower_3)

    out = Dense(30, activation='relu')(merged)
    out = Dropout(0.1)(out)
    out = Dense(10, activation='relu')(out)
    out = Dense(1, activation='sigmoid', bias_initializer=output_bias)(out)

    model = Model(input_shape, out)
    return model

def CNN_model_ensemble_expanded_more_dropout(output_bias = None):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    input_shape = Input(shape=(1000, 3, 1))
    tower_1 = Conv2D(filters=16, kernel_size=(5,1), padding='same', activation='relu')(input_shape)
    tower_1 = MaxPooling2D((2, 1), strides=(2, 1), padding='same')(tower_1)

    tower_2 = Conv2D(filters=16, kernel_size=(5,1), padding='same', activation='relu')(tower_1)
    tower_2 = MaxPooling2D((2, 1), strides=(2, 1), padding='same')(tower_2)
    tower_2 = Dropout(0.1)(tower_2)

    tower_3 =  Conv2D(filters=16, kernel_size=(5,1), padding='same', activation='relu')(tower_2)
    tower_3 = MaxPooling2D((2, 1), strides=(2, 1), padding='same')(tower_3)
    tower_3 = Dropout(0.1)(tower_3)

    tower_4 =  Conv2D(filters=12, kernel_size=(5,1), padding='same', activation='relu')(tower_3)
    tower_4 = Dropout(0.1)(tower_4)

    merged = Flatten()(tower_4)

    out = Dense(30, activation='relu')(merged)
    out = Dropout(0.1)(out)
    out = Dense(10, activation='relu')(out)
    out = Dense(1, activation='sigmoid', bias_initializer=output_bias)(out)

    model = Model(input_shape, out)
    return model

def CNN_model_ensemble_expanded(output_bias = None):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    input_shape = Input(shape=(128, 3, 1))
    tower_1 = Conv2D(filters=16, kernel_size=(5,1), padding='same', activation='relu')(input_shape)
    tower_1 = MaxPooling2D((2, 1), strides=(2, 1), padding='same')(tower_1)

    tower_2 = Conv2D(filters=16, kernel_size=(5,1), padding='same', activation='relu')(tower_1)
    tower_2 = MaxPooling2D((2, 1), strides=(2, 1), padding='same')(tower_2)

    tower_3 =  Conv2D(filters=16, kernel_size=(5,1), padding='same', activation='relu')(tower_2)
    tower_3 = MaxPooling2D((2, 1), strides=(2, 1), padding='same')(tower_3)

    merged = Flatten()(tower_3)

    out = Dense(50, activation='relu')(merged)
    out = Dropout(0.1)(out)
    out = Dense(10, activation='relu')(out)
    out = Dense(1, activation='sigmoid', bias_initializer=output_bias)(out)

    model = Model(input_shape, out)
    return model

def CNN_model_ensemble_9_channel(output_bias = None): # same as non ensemble model, used to specify with name in history
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    input_shape = Input(shape=(128, 9, 1))
    tower_1 = Conv2D(filters=16, kernel_size=(5,1), padding='same', activation='relu')(input_shape)
    tower_1 = MaxPooling2D((2, 1), strides=(2, 1), padding='same')(tower_1)

    tower_2 = Conv2D(filters=16, kernel_size=(5,1), padding='same', activation='relu')(tower_1)
    tower_2 = MaxPooling2D((2, 1), strides=(2, 1), padding='same')(tower_2)

    tower_3 =  Conv2D(filters=16, kernel_size=(5,1), padding='same', activation='relu')(tower_2)
    tower_3 = MaxPooling2D((2, 1), strides=(2, 1), padding='same')(tower_3)

    merged = Flatten()(tower_3)

    out = Dense(50, activation='relu')(merged)
    out = Dropout(0.1)(out)
    out = Dense(10, activation='relu')(out)
    out = Dense(1, activation='sigmoid', bias_initializer=output_bias)(out)

    model = Model(input_shape, out)
    return model

def CNN_model_ensemble_long_frame(output_bias = None):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    input_shape = Input(shape=(192, 3, 1))
    tower_1 = Conv2D(filters=16, kernel_size=(5,1), padding='same', activation='relu')(input_shape)
    tower_1 = MaxPooling2D((2, 1), strides=(2, 1), padding='same')(tower_1)

    tower_2 = Conv2D(filters=16, kernel_size=(5,1), padding='same', activation='relu')(tower_1)
    tower_2 = MaxPooling2D((2, 1), strides=(2, 1), padding='same')(tower_2)

    tower_3 =  Conv2D(filters=12, kernel_size=(5,1), padding='same', activation='relu')(tower_2)
    tower_3 = MaxPooling2D((2, 1), strides=(2, 1), padding='same')(tower_3)

    merged = Flatten()(tower_3)

    out = Dense(30, activation='relu')(merged)
    out = Dropout(0.1)(out)
    out = Dense(10, activation='relu')(out)
    out = Dense(1, activation='sigmoid', bias_initializer=output_bias)(out)

    model = Model(input_shape, out)
    return model

def CNN_model_2D_reg(output_bias = None):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    reg = tf.keras.regularizers.L2(l2=0.02)

    input_shape = Input(shape=(128, 9, 1))
    tower_1 = Conv2D(filters=16, kernel_size=(5,1), padding='same', activation='relu', kernel_regularizer=reg)(input_shape)
    tower_1 = MaxPooling2D((2, 1), strides=(2, 1), padding='same')(tower_1)

    tower_2 = Conv2D(filters=16, kernel_size=(5,1), padding='same', activation='relu', kernel_regularizer=reg)(tower_1)
    tower_2 = MaxPooling2D((2, 1), strides=(2, 1), padding='same')(tower_2)

    tower_3 =  Conv2D(filters=16, kernel_size=(5,1), padding='same', activation='relu', kernel_regularizer=reg)(tower_2)
    tower_3 = MaxPooling2D((2, 1), strides=(2, 1), padding='same')(tower_3)

    merged = Flatten()(tower_3)

    out = Dense(100, activation='relu')(merged)
    out = Dropout(0.2)(out)
    out = Dense(10, activation='relu')(out)
    out = Dense(1, activation='sigmoid', bias_initializer=output_bias)(out)

    model = Model(input_shape, out)
    return model

def CNN_model_2D_reg_less_filters(output_bias = None):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    reg = tf.keras.regularizers.L2(l2=0.02)

    input_shape = Input(shape=(128, 9, 1))
    tower_1 = Conv2D(filters=8, kernel_size=(5,1), padding='same', activation='relu', kernel_regularizer=reg)(input_shape)
    tower_1 = MaxPooling2D((2, 1), strides=(2, 1), padding='same')(tower_1)

    tower_2 = Conv2D(filters=12, kernel_size=(5,1), padding='same', activation='relu', kernel_regularizer=reg)(tower_1)
    tower_2 = MaxPooling2D((2, 1), strides=(2, 1), padding='same')(tower_2)

    tower_3 =  Conv2D(filters=16, kernel_size=(5,1), padding='same', activation='relu', kernel_regularizer=reg)(tower_2)
    tower_3 = MaxPooling2D((2, 1), strides=(2, 1), padding='same')(tower_3)

    merged = Flatten()(tower_3)

    out = Dense(40, activation='relu')(merged)
    out = Dropout(0.2)(out)
    out = Dense(15, activation='relu')(out)
    out = Dense(1, activation='sigmoid', bias_initializer=output_bias)(out)

    model = Model(input_shape, out)
    return model

def CNN_model_2D_simple_3d(output_bias = None):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    input_shape = Input(shape=(1000, 9, 1))
    tower_1 = Conv2D(filters=16, kernel_size=(5,1), padding='same', activation='relu')(input_shape)
    tower_1 = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(tower_1)

    tower_2 = Conv2D(filters=16, kernel_size=(5,1), padding='same', activation='relu')(tower_1)
    tower_2 = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(tower_2)

    tower_3 =  Conv2D(filters=16, kernel_size=(5,1), padding='same', activation='relu')(tower_2)
    tower_3 = MaxPooling2D((2, 1), strides=(2, 1), padding='same')(tower_3)

    merged = Flatten()(tower_3)

    out = Dense(50, activation='relu')(merged)
    out = Dropout(0.1)(out)
    out = Dense(10, activation='relu')(out)
    out = Dense(1, activation='sigmoid', bias_initializer=output_bias)(out)

    model = Model(input_shape, out)
    return model

def LSTM_model():
    input_shape = Input(shape=(128, 9))
    lstm1 = LSTM(50, dropout=0.05, return_sequences=True)(input_shape)
    lstm2 = LSTM(50, dropout=0.05)(lstm1)
    out = Dense(10, activation='relu')(lstm2)
    out = Dense(1, activation='sigmoid')(out)
    model = Model(input_shape, out)
    return model

import os
import keras
import tensorflow as tf
import numpy as np
import pandas as pd




# model = CNN_model_2D_simple()
#
# print(model.summary())

# model = CNN_model_2D_reg()
# print(model.summary())
# model = CNN_model_2D_simple_3d()
# print(model.summary())
#
# model = keras.models.load_model("models/base_train_0.0001_300_CNN_model_2022_11_17_50fc")
# print(model.summary())
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# tflite_model = converter.convert()
# with open("halfFC_11_17"+'.tflite', 'wb') as f:
#   f.write(tflite_model)
# print(tf.lite.experimental.Analyzer.analyze(model_content=tflite_model))

# val_st = np.array(  pd.read_csv('val_set.csv')).reshape(-1,128,3)
# val_target = np.array(  pd.read_csv('val_set_target.csv'))
#
# print(val_st.shape)
# print(val_target.shape)
#
# MyList = np.array([2, 2, 2, 4, 5, 5, 5, 7, 8, 8, 10, 12])
# print(np.count_nonzero(MyList == 2))
