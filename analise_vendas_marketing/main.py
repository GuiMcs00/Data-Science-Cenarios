import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def calcular_covariancia(x, y):
    med_x = np.mean(x)
    med_y = np.mean(y)

    if len(x) == len(y):
        covariancia = sum((x_i - med_x) * (y_i - med_y) for x_i, y_i in zip(x,y)) / len(x) - 1
    else:
        raise TypeError("Os vetores não são compatíveis")

    return covariancia


def calcular_correlacao(x, y):
    covariancia = calcular_covariancia(x, y)

    correlacao = covariancia / (np.std(x) * np.std(y))
    return correlacao


dados = pd.read_csv('vendas_marketing.csv')

estatisticas = dados.describe()

estatisticas.to_csv('output_estatisticas.csv')

mkt_vendas = dados.drop(columns=['Receita'])


facebook_vector = np.array(mkt_vendas['Gastos em Marketing (Facebook Ads)'])
google_vector = np.array(mkt_vendas['Gastos em Marketing (Google Ads)'])
instagram_vector = np.array(mkt_vendas['Gastos em Marketing (Instagram Ads)'])
vendas_vector = np.array(mkt_vendas['Vendas'])
visitas_vector = np.array(mkt_vendas['Visitas ao Site'])

facebook_vendas = calcular_correlacao(facebook_vector, vendas_vector)
google_vendas = calcular_correlacao(google_vector, vendas_vector)
instagram_vendas = calcular_correlacao(instagram_vector, vendas_vector)
visitas_vendas = calcular_correlacao(visitas_vector, vendas_vector)

# Exibir correlações
print("Correlação Facebook Ads e Vendas:", facebook_vendas)
print("Correlação Google Ads e Vendas:", google_vendas)
print("Correlação Instagram Ads e Vendas:", instagram_vendas)
print("Correlação Visitas ao Site e Vendas:", visitas_vendas)


# Gráfico de dispersão: Gastos em Marketing (Google Ads) vs Vendas
plt.figure(figsize=(10, 6))
sns.scatterplot(data=dados, x='Gastos em Marketing (Facebook Ads)', y='Vendas')
plt.title('Gastos em Marketing (Facebook Ads)')
plt.show()

# Gráfico de dispersão: Gastos em Marketing (Google Ads) vs Vendas
plt.figure(figsize=(10, 6))
sns.scatterplot(data=dados, x='Gastos em Marketing (Google Ads)', y='Vendas')
plt.title('Gastos em Marketing (Google Ads) vs Vendas')
plt.show()

# Gráfico de dispersão: Gastos em Marketing (Google Ads) vs Vendas
plt.figure(figsize=(10, 6))
sns.scatterplot(data=dados, x='Gastos em Marketing (Instagram Ads)', y='Vendas')
plt.title('Gastos em Marketing (Instagram Ads)')
plt.show()

# Gráfico de dispersão: Visitas ao Site vs Vendas
plt.figure(figsize=(10, 6))
sns.scatterplot(data=dados, x='Visitas ao Site', y='Vendas')
plt.title('Visitas ao Site vs Vendas')
plt.show()

