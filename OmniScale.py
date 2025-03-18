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
file_path = "datasetBitcoin.csv"  # Certifique-se de que o arquivo está no mesmo diretório
dataset_name = os.path.splitext(os.path.basename(file_path))[0]
df = pd.read_csv(file_path, parse_dates=['Date'], dayfirst=True, index_col='Date')

# Converter a coluna 'Price' para float
df['Price'] = df['Price'].astype(str).str.replace(',', '').astype(float)

# Ordenar os dados por data
df = df.sort_index()

# ===============================================
# PREPARAR OS DADOS PARA A REDE NEURAL
# ===============================================
history = 30
horizon = 1
train_ind = int(len(df) * 0.8)
train = df[:train_ind]
test = df[train_ind:]

print('Training size:', train.shape[0])
print('Test size:', test.shape[0])

input_features = ['Price']
data = df[input_features].values

x_data, y_data = [], []
for i in range(len(data) - history - horizon + 1):
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
dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=32, num_workers=0)

# Criar e treinar o modelo
model = create_model(OmniScaleCNN, d=False, dls=dls).to(device)
learn = Learner(dls, model, metrics=[mae, rmse], opt_func=Adam)
learn.fit_one_cycle(3, lr_max=0.01)

# Salvar gráfico de métricas localmente
learn.plot_metrics()
fig = plt.gcf()
fig.savefig(f"{resultados_path}/{dataset_name}_metrics.png", bbox_inches='tight', pad_inches=0.1)
plt.show()

# ===============================================
# TESTE DO MODELO E AVALIAÇÃO FINAL
# ===============================================
test_ds = dls.valid.dataset.add_test(X_test, y_test)
test_dl = dls.valid.new(test_ds)
test_preds = learn.get_preds(dl=test_dl, with_decoded=True)[2]

y_pred = test_preds.numpy().squeeze()
y_true = y_test.squeeze()

# Salvar previsões e métricas
pickle.dump(y_pred, open(f"{resultados_path}/{dataset_name}_Pred.pkl", 'wb'))
pickle.dump(y_true, open(f"{resultados_path}/{dataset_name}_True.pkl", 'wb'))

print("Arquivos salvos:", os.listdir(resultados_path))
