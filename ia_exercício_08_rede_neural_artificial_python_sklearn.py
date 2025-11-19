import pandas as pd
# Carregando dados do arquivo CSV
url = 'https://raw.githubusercontent.com/LeandroMCarv/predicao-diabetes-perceptron/refs/heads/main/diabetes.csv'
base_Treinamento = pd.read_csv(url,sep=';', encoding = 'latin1')
base_Treinamento_df = pd.read_csv(url,sep=';', encoding = 'latin1').values
print("---------------------------------")
print("Dados dos Pacientes - TREINAMENTO")
print("---------------------------------")
print(base_Treinamento)
print("---------------------------------")

# Extração dos Atributos a serem utilizadas pela rede
print("Atributos de Entrada")
print("---------------------------------")
print(base_Treinamento_df[:, 1:8])

print("----------------------------")
print("Classificação Supervisionada")
print("----------------------------")
print(base_Treinamento_df[:, 8])

"""### Pré-processamento de Dados"""

import numpy as np
from sklearn import preprocessing

X = base_Treinamento.drop("Outcome", axis=1)
y = base_Treinamento["Outcome"]

scaler = preprocessing.StandardScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2
)

#Concatenação de Atributos (Colunas)
print("--------------------------------")
print("Atributos de Entrada - Numéricos")
print("--------------------------------")
print(X_train)

print("----------------------------------------")
print("Classificação Supervisionada - Numéricos")
print("----------------------------------------")
print(y_train)

"""### Treinamento do Neurônio Perceptron"""

from sklearn.linear_model import Perceptron
# Treinamento do Perceptron a partir dos atributos de entrada e classificações

modelo = Perceptron()
modelo.fit(X_train, y_train)

# Acurácia do modelo, que é : 1 - (predições erradas / total de predições)
# Acurácia do modelo: indica uma performance geral do modelo.
# Dentre todas as classificações, quantas o modelo classificou corretamente;
# (VP+VN)/N

print('Acurácia: %.3f' % modelo.score(X_train, y_train))

"""### ----------------------------------------------------------------------------

# Validação do Aprendizado

### Predição Simples
"""

y_pred = modelo.predict(X_test)

print("Predições:", y_pred.tolist())
print("Reais:     ", y_test.tolist())

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Acurácia no TESTE
print("\nAcurácia (TESTE): %.3f" % accuracy_score(y_test, y_pred))

# Relatório completo
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

# Matriz de confusão
print("\nMatriz de Confusão:")
print(confusion_matrix(y_test, y_pred))

# Reverte o StandardScaler
X_train_original = scaler.inverse_transform(X_train)
X_test_original  = scaler.inverse_transform(X_test)

# Reconstrói DataFrames
colunas = X.columns
X_train_df = pd.DataFrame(X_train_original, columns=colunas)
X_test_df  = pd.DataFrame(X_test_original,  columns=colunas)

# Converte y e reseta índices
y_train_df = y_train.reset_index(drop=True).to_frame(name="Outcome")
y_test_df  = y_test.reset_index(drop=True).to_frame(name="Outcome")

# Zera índices dos X também
X_train_df = X_train_df.reset_index(drop=True)
X_test_df  = X_test_df.reset_index(drop=True)

# Concatena corretamente
train_df = pd.concat([X_train_df, y_train_df], axis=1)
test_df  = pd.concat([X_test_df,  y_test_df],  axis=1)

# Salva
train_df.to_csv("treinamento.csv", index=False,sep=';')
test_df.to_csv("teste.csv", index=False,sep=';')
