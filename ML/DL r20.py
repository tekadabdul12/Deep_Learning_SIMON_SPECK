import numpy as np
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from hyperopt import fmin, tpe, hp, Trials

# 1. Siapkan dataset
# x = np.array([1,2,3,4,5])
# y = np.array([3,6,9,12,15])

y = np.array([0x065b,0x065b]) #output

x = np.array([[0x444e51c4, 0x42b3a873, 0x8d2357f8, 0x9304244f, 0xa9c2a3a1,
               0xbc20a995, 0x49f4bcdb, 0xfadd4021, 0x78ebe0cf, 0x6d625609,
               0x2a5a93e9, 0x02b34f62, 0x92e5110b, 0x1f92ce58, 0x019116ac,
               0x46395074, 0x52c78e83, 0xd2f4dc5d, 0x3358ca4c, 0xcab6bc6f,
               0xa5733197, 0x892d6a2a, 0xc2238915, 0x14b3eac9, 0x469fe78e, 0x46873572],
              [0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4A,
                0x4B, 0x4C, 0x4D, 0x4E, 0x4F, 0x50, 0x51, 0x52, 0x53, 0x54,
                0x55, 0x56, 0x57, 0x58, 0x59, 0x5A]])
# 2. Buat model
def create_model(params):
    model = keras.Sequential()
    model.add(keras.layers.Dense(params['units'][0], input_dim=26, activation=params['activation']))
    for i in range(1, params['num_layers']):
        model.add(keras.layers.Dense(params['units'][i], activation=params['activation']))
    model.add(keras.layers.Dense(1))
    model.compile(loss='mse', optimizer=params['optimizer'](learning_rate=params['learning_rate']))
    return model

# 3. Tentukan rentang nilai hyperparameter
space = {
    'num_layers': hp.choice('num_layers', range(1, 6)),
    'units': [hp.quniform('units_'+str(i), 4, 512, 4) for i in range(6)],
    'activation': hp.choice('activation', ['relu']),
    'optimizer': hp.choice('optimizer', [keras.optimizers.Adam,keras.optimizers.Adamax,keras.optimizers.Adadelta]),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.0001), np.log(0.01))
}

# 4. Buat fungsi objektif
def objective(params):
    model = create_model(params)
    early_stop = EarlyStopping(monitor='loss', patience=50, restore_best_weights=True)
    history = model.fit(x, y, epochs=1000, validation_split=0.2, callbacks=[early_stop])
    val_loss = history.history['loss'][-1]

    data_loss = history.history['loss']
    smallest_value = min(data_loss)
    smallest_index = data_loss.index(smallest_value)
    print(smallest_index, smallest_value)
    print(data_loss)

    print(history)
    return {'loss': smallest_value, 'status': 'ok', 'params':params}

# 5. Jalankan Bayesian optimizer
trials = Trials()
best = fmin(objective, space, algo=tpe.suggest, max_evals=20, trials=trials, return_argmin=True)
print("best result",best)



for i, result in enumerate(trials.results):
    print("Iteration ", i+1)
    print(result)

