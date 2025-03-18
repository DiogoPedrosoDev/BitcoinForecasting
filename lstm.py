import os
import warnings
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
# CONFIGURAR PASTA DE RESULTADOS
# ===============================================
resultados_path = "Resultados"
os.makedirs(resultados_path, exist_ok=True)

# ===============================================
# CONFIGURAR GPU PARA TREINAMENTO RÁPIDO
# ===============================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Rodando em:", device)

# ===============================================
# CARREGAR O DATASET LOCALMENTE
# ===============================================
file_path = "datasetBitcoin.csv"  # Arquivo no mesmo diretório do código
dataset_name = os.path.splitext(os.path.basename(file_path))[0]
df = pd.read_csv(file_path, parse_dates=['Date'], dayfirst=True, index_col='Date')

# Converter a coluna 'Price' para float (caso tenha separador de milhares)
df['Price'] = df['Price'].astype(str).str.replace(',', '').astype(float)

# Ordenar os dados por data
df = df.sort_index()

# ===============================================
# DEFINIR PARÂMETROS DO MODELO
# ===============================================
history = 30  # Número de dias passados para prever o futuro
horizon = 1   # Quantidade de dias a serem previstos
max_evals = 50  # Número de testes de otimização de hiperparâmetros

model_name = 'LSTM_Bitcoin'

# Variável alvo (Preço do Bitcoin)
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

# Ajustar formato dos dados
x_data = np.swapaxes(x_data, 1, 2)

# Separar treino, validação e teste
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
"""
search_space = {
    'batch_size': hp.choice('bs', [16, 32, 64, 128]),
    "lr": hp.choice('lr', [0.01, 0.001, 0.0001]),
    "epochs": hp.choice('epochs', [20, 50, 100]),
    "patience": hp.choice('patience', [5, 10]),
    "optimizer": hp.choice('optimizer', [Adam]),
    "n_layers": hp.choice('n_layers', [1, 2, 3, 4, 5]),
    "hidden_size": hp.choice('hidden_size', [50, 100, 200]),
    "bidirectional": hp.choice('bidirectional', [True, False])
}
"""
search_space = {
    'batch_size': hp.choice('bs', [32]),  # Usar apenas um valor fixo para evitar otimização
    "lr": hp.choice('lr', [0.01]),  # Fixa um valor rápido
    "epochs": hp.choice('epochs', [3]),  # Apenas 3 épocas para testar rapidamente
    "patience": hp.choice('patience', [1]),  # Para early stopping mais rápido
    "optimizer": hp.choice('optimizer', [Adam]),
    "n_layers": hp.choice('n_layers', [1]),  # Apenas 1 camada para reduzir o tempo de treino
    "hidden_size": hp.choice('hidden_size', [50]),  # Menos neurônios para ser mais leve
    "bidirectional": hp.choice('bidirectional', [False])  # Desativa bidirecionalidade para simplificar
}

# ===============================================
# TREINAMENTO E OTIMIZAÇÃO
# ===============================================
def create_model_hypopt(params):
    try:
        gc.collect()
        batch_size = params["batch_size"]
        dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[batch_size, batch_size], num_workers=0)

        model = create_model(LSTM, d=False, dls=dls).to(device)
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
model = create_model(LSTM, d=False, dls=dls).to(device)

learn = Learner(dls, model, metrics=[mae, rmse], opt_func=params['optimizer'])
learn.fit_one_cycle(params['epochs'], lr_max=params['lr'],
                    cbs=EarlyStoppingCallback(monitor='valid_loss', patience=params['patience']))

# ===============================================
# SALVAR GRÁFICO DE MÉTRICAS LOCALMENTE (Correção para imagem branca)
# ===============================================

# Gerar o gráfico de métricas
learn.plot_metrics()

# Capturar a figura ATUAL do Matplotlib
fig = plt.gcf()  # Obtém a figura gerada pelo Matplotlib

# Definir nome dos arquivos de saída incluindo o nome do dataset
output_file_pdf = f"{resultados_path}/{model_name}_{dataset_name}_metrics.pdf"
output_file_png = f"{resultados_path}/{model_name}_{dataset_name}_metrics.png"

# Salvar ANTES de exibir para evitar que plt.show() limpe a figura
fig.savefig(output_file_pdf, bbox_inches='tight', pad_inches=0.1)
fig.savefig(output_file_png, bbox_inches='tight', pad_inches=0.1)

# Agora exibir o gráfico
plt.show()
# ===============================================
# TESTE DO MODELO E AVALIAÇÃO FINAL
# ===============================================
test_ds = dls.valid.dataset.add_test(X_test, y_test)
test_dl = dls.valid.new(test_ds)
test_preds = learn.get_preds(dl=test_dl, with_decoded=True)[2]

y_pred = test_preds.numpy().squeeze()
y_true = y_test.squeeze()

# Calcular métricas
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)

print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")

# Definir nomes dos arquivos de saída incluindo o nome do dataset
pred_file = f"{resultados_path}/{model_name}_{dataset_name}_Pred.pkl"
true_file = f"{resultados_path}/{model_name}_{dataset_name}_True.pkl"

# Salvar previsões e métricas localmente
pickle.dump(y_pred, open(pred_file, 'wb'))
pickle.dump(y_true, open(true_file, 'wb'))

# Verificar arquivos salvos
print("Arquivos salvos:")
print(f"- {output_file_pdf}")
print(f"- {output_file_png}")
print(f"- {pred_file}")
print(f"- {true_file}")
print("Lista completa de arquivos na pasta 'Resultados':", os.listdir(resultados_path))
