import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
from sklearn.preprocessing import  MinMaxScaler

model = tf.keras.Sequential(
    [

        keras.layers.Dense(600, activation='relu', input_shape=(None,26)),
        keras.layers.Dense(600, activation='relu'),
        keras.layers.Dense(600, activation='relu'),
        keras.layers.Dense(600, activation='relu'),
        keras.layers.Dense(600, activation='relu'),
        keras.layers.Dense(600, activation='relu'),
        keras.layers.Dense(600, activation='relu'),
        keras.layers.Dense(600, activation='relu'),
        keras.layers.Dense(600, activation='relu'),
        keras.layers.Dense(600, activation='relu'),


        keras.layers.Dense(1)

    ]
)

model.compile( loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

# xs = np.array([0b0010,0b0011,0b0100,0b0101,0b0110], dtype=float) #input = 2,3,4,5,6
#
# ys = np.array([0b0100,0b0110,0b1000,0b1010,0b1100], dtype=float) #output = 4,6,8,10,12

#output
ys = np.array([
    0xbb7c,0xbb7c
])

#input
xs = np.array([[0xd43517f5, 0x1fd170b7, 0x2b57196d, 0x27d57a40, 0x78253751,
               0x6986efcb, 0x4c7e75c6, 0xc7fa8c9c, 0x86a272b3, 0x9b8d4d17,
               0x83268570, 0x57bbe230, 0x3201a334, 0x908add30, 0xf7b87dca,
               0x24b2623a, 0xf20beea, 0x92ff55d1, 0x3e0395aa, 0x051947e2,
               0x39146983, 0x5e735b8d, 0xd8f51969, 0xa955f167, 0x445c82bf, 0x8cdcf153],
              [0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4A,
                0x4B, 0x4C, 0x4D, 0x4E, 0x4F, 0x50, 0x51, 0x52, 0x53, 0x54,
                0x55, 0x56, 0x57, 0x58, 0x59, 0x5A]])


print(np.shape(xs))
print(np.shape(ys))
print(xs, xs.dtype)
print(ys, xs.dtype)

class haltCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('loss') <= 0.0001):
            print("\n\n\nReached 0.05 loss value so cancelling training!\n\n\n")

            self.model.stop_training = True

stop = haltCallback()

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=200)

fit1 = model.fit(xs,ys, epochs=10000, callbacks=[stop])

#result = model.predict(np.array([0b1111],dtype=float)) #predic = 7
# print(result)
# result = np.round(result)

# print(result)
#
#
# for i in result:
#     for a in i:
#         print(a)
#     a = np.array(a,dtype=int)
#
# print(a.dtype , a, a.shape)
# print(np.binary_repr(a))

