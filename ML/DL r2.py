import numpy as np
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from hyperopt import fmin, tpe, hp, Trials
import csv

# 1. Siapkan dataset
# x = np.array([1,2,3,4,5])
# y = np.array([3,6,9,12,15])

data = []
with open('../output_simon.csv', mode='r', newline='') as file:
    reader = csv.DictReader(file)
    for row in reader:
        data.append(row["cipher"])  # Menambahkan elemen kolom "Nama" ke list

data = [int(x, 16) for x in data]



data_input = []
with open('../input.csv', mode='r', newline='') as file:
    reader = csv.DictReader(file)
    for row in reader:
        data_input.append(row["Plaintext"])  # Menambahkan elemen kolom "Nama" ke list

data_input = [int(x, 16) for x in data_input]

y = np.array([0x0908,0x0908]) #output

x = np.array([data,
              data_input])

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

