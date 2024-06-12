import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np
from scipy import stats
df = pd.read_csv (r'C:\Users\jacso\Desktop\Homework\XLM-USD-Daily-09.15.14-10.30.21.csv')
df=df.dropna()
df.reset_index(drop=True, inplace=True)
print(df[df["Open"].isna()])


corrMatrix = df.corr()
sn.heatmap(corrMatrix, annot=True)
print(df)
X = pd.DataFrame(df, columns= ['Low'])
y = pd.DataFrame(df, columns= ['High'])


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
model = LinearRegression().fit(X, y)
y_pred= model.predict(X)
r_sq= model.score(X, y)
print('coefR^2:', r_sq)
print (y_test)
regressor=LinearRegression()
regressor.fit(X_train,y_train)
print(regressor.intercept_)
print(regressor.coef_)
#compararea valorilor reale cu cele prezise
y_pred=regressor.predict(X_test)

plt.plot(X,y,'o', color='red')
plt.plot(X, regressor.coef_*X + regressor.intercept_)

#CalculuiErorilor
print('Eroarea medie absoluta:',metrics.mean_absolute_error(y_test,y_pred))#MAE
print('Eroare medie patratica:',metrics.mean_squared_error(y_test,y_pred))#MSE
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
