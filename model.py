
from keras.layers import Input, Conv2D, UpSampling2D, BatchNormalization, Dense
from keras.layers import Flatten, Activation
from keras.models import Model
from keras import regularizers
from keras.initializers import glorot_normal

def encoder_model(reg=0.1):

    x_input = Input(shape=(28, 28, 1))
    x = Conv2D(filters=16, kernel_size=3, strides=2, padding='same',
               activation='relu',
               kernel_regularizer=regularizers.l2(reg),
               kernel_initializer=glorot_normal())(x_input)
    x = Conv2D(filters=8, kernel_size=3, strides=2, padding='same',
               activation='relu',
               kernel_regularizer=regularizers.l2(reg),
               kernel_initializer=glorot_normal())(x)
    x = Conv2D(filters=8, kernel_size=3, strides=2, padding='same',
               activation='relu',
               kernel_regularizer=regularizers.l2(reg),
               kernel_initializer=glorot_normal())(x)
    x = UpSampling2D(2)(x)
    x = Conv2D(filters=4, kernel_size=3, padding='same',
               activation='sigmoid',
               kernel_regularizer=regularizers.l2(reg),
               kernel_initializer=glorot_normal())(x)

    model = Model(x_input, x, name='encoder')
    return model


def decoder_model(reg=0.02):
    x_input = Input(shape=(8, 8, 4))
    x = Conv2D(filters=8, kernel_size=3, strides=2, padding='same',
               activation='relu',
               kernel_regularizer=regularizers.l2(reg),
               kernel_initializer=glorot_normal())(x_input)
    x = UpSampling2D(2)(x)
    x = Conv2D(filters=8, kernel_size=3, padding='same',
               activation='relu',
               kernel_regularizer=regularizers.l2(reg),
               kernel_initializer=glorot_normal())(x)
    x = UpSampling2D(2)(x)
    x = Conv2D(filters=16, kernel_size=3,
               activation='relu',
               kernel_regularizer=regularizers.l2(reg),
               kernel_initializer=glorot_normal())(x)
    x = UpSampling2D(2)(x)
    x = Conv2D(filters=1, kernel_size=3, padding='same',
               activation='sigmoid',
               kernel_regularizer=regularizers.l2(reg),
               kernel_initializer=glorot_normal())(x)
    model = Model(x_input, x, name='decoder')
    return model

def classifier_model(reg=1e-2, dropout=0.1):

    # convolution layer
    x_input = Input(shape=(28, 28, 1), name='light_curve_input')
    x = Conv2D(filters=8,
                  kernel_size=5,
                  strides=2,
                  padding='same',
                  kernel_regularizer=regularizers.l2(reg),
                  kernel_initializer=glorot_normal())(x_input)
    x = BatchNormalization(axis=2)(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=16,
                  kernel_size=5,
                  strides=2,
                  padding='same',
                  kernel_regularizer=regularizers.l2(reg),
                  kernel_initializer=glorot_normal())(x)
    x = BatchNormalization(axis=2)(x)
    x = Activation('relu')(x)
    x = Flatten()(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(10,
              kernel_regularizer=regularizers.l2(reg),
              activation='sigmoid',
              name='output')(x)

    model = Model(inputs=x_input, outputs=x, name='PhNet')
    model.compile(optimizer='adam' , loss='categorical_crossentropy', metrics=['accuracy'])

    return model