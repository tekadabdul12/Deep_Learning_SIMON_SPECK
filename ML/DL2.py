import numpy as np
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from hyperopt import fmin, tpe, hp, Trials

# 1. Siapkan dataset
# x = np.array([1,2,3,4,5])
# y = np.array([3,6,9,12,15])

y = np.array([0x4748,0x4748]) #output

x = np.array([[0x47090, 0x470a0, 0x470b0, 0x470c0, 0x470d0,
               0x470e0, 0x470f0, 0x47000, 0x47010, 0x47020,
               0x47030, 0x47040, 0x47050, 0x47060, 0x47070,
               0x47180, 0x47190, 0x471a0, 0x471b0, 0x471c0,
               0x471d0, 0x471e0, 0x471f0, 0x47100, 0x47110, 0x47120],
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
    'optimizer': hp.choice('optimizer', [keras.optimizers.Adam,keras.optimizers.Adadelta,keras.optimizers.Adamax]),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.0001), np.log(0.01))
}

# 4. Buat fungsi objektif
def objective(params):
    model = create_model(params)
    early_stop = EarlyStopping(monitor='loss', patience=100, restore_best_weights=True)
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

