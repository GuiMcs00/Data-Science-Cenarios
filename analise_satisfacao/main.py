import pandas as pd
import numpy as np

csv_satisfacao = pd.read_csv('satisfacao_clientes.csv')
valores = csv_satisfacao.drop(columns=['Cliente'])


estatisticas = pd.DataFrame(
    {'Média': valores.mean(),
     'Mediana': valores.median(),
     'Desvio Padrão': valores.std()}
)
estatisticas = estatisticas.T

print(estatisticas)
estatisticas.to_csv('estatisticas.csv')
