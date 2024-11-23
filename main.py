import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import shapiro, pearsonr, spearmanr
import statsmodels as sm 

#normalidade
def test_normality(data, column):
    p_value = shapiro(data[column])
    return p_value

cons_cerveja = pd.read_csv("consumo_cerveja.csv")
mtcars = pd.read_csv("mtcars.csv")

columns = ['mpg', 'cyl', 'disp', 'hp']
for col in columns:
    p_val = test_normality(mtcars, col)
    print(f"Teste de normalidade para {col}: valor-p ={p_val}.")

#scatterplot

mpg_ = mtcars['mpg']
cyl_ = mtcars['cyl']
disp_ = mtcars['disp']
hp_ = mtcars['hp']
filtros = [mpg_, cyl_, disp_, hp_]


plt.scatter(mpg_, cyl_)
plt.scatter(mpg_, disp_)
plt.scatter(mpg_, hp_)

plt.title('Scatterplot')

plt.show()

#histograma
for i in range(4):
    sns.histplot(filtros[i])
    plt.show()

#coef de correlação
for i in range(1, 4):
    coef = spearmanr(filtros[0], filtros[i])
    print(f"coeficiente {i}: {coef}")

# regressao liner 
model = sm.LinearRegression()
for i in range(1, 4):
    result = model.fit(filtros[0], filtros[i])
    print(f"Regressão {i}: {result}")
