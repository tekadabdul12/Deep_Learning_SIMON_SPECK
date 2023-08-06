import numpy as np
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from hyperopt import fmin, tpe, hp, Trials

# 1. Siapkan dataset
x = np.array([1,2,3,4,5])
y = np.array([3,6,9,12,15])

# 2. Buat model
def create_model(units=16,units2=16, activation='relu', optimizer='adam'):
    model = keras.Sequential([
        keras.layers.Dense(units=units, activation=activation, input_dim=1),
        keras.layers.Dense(units=units2, activation=activation),
        keras.layers.Dense(units=1)
    ])
    model.compile(loss='mse', optimizer=optimizer)
    return model

# 3. Tentukan rentang nilai hyperparameter
space = {
    'units': hp.quniform('units', 4, 64, 4),
    'units2': hp.quniform('units2', 4, 64, 4),
    'activation': hp.choice('activation', ['tanh','relu','sigmoid']),
    'optimizer': hp.choice('optimizer', ['adam', 'sgd'])

}

# 4. Buat fungsi objektif
def objective(params):
    model = create_model(**params)
    early_stop = EarlyStopping(monitor='val_loss', patience=20)
    history = model.fit(x, y, epochs=100, validation_split=0.2, callbacks=[early_stop])
    val_loss = history.history['loss'][-1]
    print(history)
    return {'loss': val_loss, 'status': 'ok'}

# 5. Jalankan Bayesian optimizer
trials = Trials()
best = fmin(objective, space, algo=tpe.suggest, max_evals=2, trials=trials)
print(best)
