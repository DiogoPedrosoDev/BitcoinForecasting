import os
import gc
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import time

from tsai.all import *
from fastai.callback.tracker import EarlyStoppingCallback
from sklearn.metrics import mean_squared_error, mean_absolute_error
from hyperopt import Trials, STATUS_OK, STATUS_FAIL, tpe, fmin, hp, space_eval

# ===============================================
# CONFIGURAR GPU PARA TREINAMENTO RÁPIDO
# ===============================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Rodando em:", device)

# ===============================================
# CARREGAR O DATASET LOCALMENTE
# ===============================================
file_path = "datasetBitcoin.csv"  # Substitua pelo caminho correto

df = pd.read_csv(file_path, parse_dates=['Date'], dayfirst=True, index_col='Date')

# Converter a coluna 'Price' para float (caso tenha separador de milhares)
df['Price'] = df['Price'].astype(str).str.replace(',', '').astype(float)

# Ordenar os dados por data
df = df.sort_index()

# ===============================================
# DEFINIR PARÂMETROS DO MODELO
# ===============================================
history = 24
horizon = 1
test_ratio = 0.2
max_evals = 50

model_name = 'ResCNN_Bitcoin'
input_features = ['Price']
data = df[input_features].values

# ===============================================
# SEPARAR DADOS EM TREINO E TESTE
# ===============================================
train_ind = int(len(df) * 0.8)
train = df[:train_ind]
test = df[train_ind:]

print('Training size:', train.shape[0])
print('Test size:', test.shape[0])

# ===============================================
# PREPARAR OS DADOS PARA A REDE NEURAL
# ===============================================
x_data, y_data = [], []
length = data.shape[0]

for i in range(length - history - horizon + 1):
    x = data[i:i+history, :]
    y = data[i+history:i+history+horizon, 0]
    x_data.append(x)
    y_data.append(y)

x_data = np.array(x_data)
y_data = np.array(y_data)

x_data = np.swapaxes(x_data, 1, 2)

test_length = len(test) - horizon + 1
train_valid_length = x_data.shape[0] - test_length
train_length = int(train_valid_length * 0.8)
valid_length = train_valid_length - train_length

X_train = x_data[:train_length]
y_train = y_data[:train_length]
X_valid = x_data[train_length:train_valid_length]
y_valid = y_data[train_length:train_valid_length]
X_test = x_data[train_valid_length:]
y_test = y_data[train_valid_length:]

# ===============================================
# CONFIGURAÇÃO DOS DADOS PARA O MODELO
# ===============================================
X, y, splits = combine_split_data([X_train, X_valid], [y_train, y_valid])
tfms = [None, [TSRegression()]]
dsets = TSDatasets(X, y, tfms=tfms, splits=splits, inplace=True)

# ===============================================
# OTIMIZAÇÃO DOS HIPERPARÂMETROS
# ===============================================
search_space = {
    'batch_size': hp.choice('bs', [16, 32, 64, 128]),
    "lr": hp.choice('lr', [0.01, 0.001, 0.0001]),
    "epochs": hp.choice('epochs', [20, 50, 100]),
    "patience": hp.choice('patience', [5, 10]),
    "optimizer": hp.choice('optimizer', [Adam])
}

# ===============================================
# TREINAMENTO E OTIMIZAÇÃO
# ===============================================
def create_model_hypopt(params):
    try:
        gc.collect()
        batch_size = params["batch_size"]
        dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[batch_size, batch_size], num_workers=0)

        model = create_model(ResCNN, d=False, dls=dls).to(device)
        learn = Learner(dls, model, metrics=[mae, rmse], opt_func=params['optimizer'])
        learn.fit_one_cycle(params['epochs'], lr_max=params['lr'],
                            cbs=EarlyStoppingCallback(monitor='valid_loss', patience=params['patience']))
        val_loss = learn.recorder.values[-1][1]
        return {'loss': val_loss, 'status': STATUS_OK}

    except:
        return {'loss': None, 'status': STATUS_FAIL}

trials = Trials()
best = fmin(create_model_hypopt, space=search_space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
params = space_eval(search_space, best)

# ===============================================
# TREINAMENTO FINAL DO MODELO
# ===============================================
batch_size = params["batch_size"]
dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[batch_size, batch_size], num_workers=0)
model = create_model(ResCNN, d=False, dls=dls).to(device)

learn = Learner(dls, model, metrics=[mae, rmse], opt_func=params['optimizer'])
learn.fit_one_cycle(params['epochs'], lr_max=params['lr'],
                    cbs=EarlyStoppingCallback(monitor='valid_loss', patience=params['patience']))

# Salvar gráfico de métricas localmente
os.makedirs("Resultados", exist_ok=True)
learn.plot_metrics()
plt.savefig("Resultados/ResCNN_Bitcoin_metrics.pdf", bbox_inches='tight', pad_inches=0.1)

# ===============================================
# TESTE DO MODELO E AVALIAÇÃO FINAL
# ===============================================
test_ds = dls.valid.dataset.add_test(X_test, y_test)
test_dl = dls.valid.new(test_ds)
test_preds = learn.get_preds(dl=test_dl, with_decoded=True)[2]

y_pred = test_preds.numpy().squeeze()
y_true = y_test.squeeze()

# Salvar métricas
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)

print("RMSE:", rmse)
print("MAE:", mae)

# Salvar previsões e métricas localmente
pickle.dump(y_pred, open("Resultados/ResCNN_Bitcoin_Pred.pkl", 'wb'))
pickle.dump(y_true, open("Resultados/ResCNN_Bitcoin_True.pkl", 'wb'))