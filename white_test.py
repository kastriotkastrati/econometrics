from statsmodels.stats.diagnostic import het_white
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import pandas as pd

url = 'https://raw.githubusercontent.com/kastriotkastrati/econometrics/main/Minitab.csv'

data = pd.read_csv(url)
data.info()

y = data['PRICE']

x = data[['AGE', 'SIZE']]
x = sm.add_constant(x)
model = sm.OLS(y, x).fit()



white_test = het_white(model.resid,  model.model.exog)

labels = ['x^2', 'p-value', 'F-Statistic', 'F-Test p-value']


print(dict(zip(labels, white_test)))
