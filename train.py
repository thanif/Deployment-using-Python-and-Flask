import pandas as pd
from sklearn import linear_model
from sklearn.externals import joblib

df = pd.read_csv('data.csv')

x = df[['sqft_living']]

y = df[['price']]

reg = linear_model.LinearRegression()

reg.fit(x,y)

print (reg.score(x,y))

print (reg.predict([[3300]]))

joblib.dump(reg,"linear_reg.pkl")
